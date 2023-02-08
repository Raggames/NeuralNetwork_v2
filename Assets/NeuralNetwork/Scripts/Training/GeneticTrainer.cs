using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    /// <summary>
    /// Achieved 85% accuracy on function regression with 20000 epochs
    /// </summary>
    /// 
    public class GeneticTrainer : NeuralNetworkTrainer
    {
        [Header("---- GeneticTrainer -----")]

        public ModelBuilder Builder;
        public List<NeuralNetwork> currentNetworks = new List<NeuralNetwork>();

        [Header("---- PARAMETERS -----")]
        ///The maximum of iterations without finding any better set than Epoch_Best_Weigths
        public int BestSetAutoStop = 1000;
        public int LowerBestAccuracyThreshold = 500;
        // Run is considered as validated if heuristic under HeuristicThreshold
        public float HeuristicThreshold = 0.01f;
        public int GenerationSize = 10;

        public float MutationRate = 0.001f;
        public float MutationChancesPurcent = 3;

        [Header("----RUNTIME----")]
        public float[] Heuristics;
        public double[] Accuracies;
        public double[,] AccuraciesData;

        public int ResetToBestCount = 0;
        public int LowerBestAccuracyCount = 0;

        private void Start()
        {
            CreateNetworks();
            Train();
        }

        private void CreateNetworks()
        {
            currentNetworks.Clear();

            for (int i = 0; i < GenerationSize; ++i)
            {
                currentNetworks.Add(new NeuralNetwork());
            }

            for (int i = 0; i < GenerationSize; ++i)
            {
                currentNetworks[i].CreateNetwork(this, Builder);
            }
        }

        public virtual async void Train()
        {
            Training_Best_Mean_Error = -1;
            TrainingSetting.Init();

            InitializeTrainingBestWeightSet(currentNetworks[0]);

            AccuraciesData = new double[GenerationSize, 3];
            Accuracies = new double[GenerationSize];

            for (int i = 0; i < Epochs; ++i)
            {
                CurrentEpoch = i;

                double[] input_values;
                double[] test_values;
                TrainingSetting.GetNextValues(out input_values, out test_values);

                Task<double[]>[] tasks = new Task<double[]>[GenerationSize];
                CancellationToken[] tokens = new CancellationToken[GenerationSize];

                double[][] all_results = new double[GenerationSize][];

                for (int g = 0; g < GenerationSize; ++g)
                {
                    tokens[g] = new CancellationToken();
                    NeuralNetwork current = currentNetworks[g];

                    tasks[g] = Task<double[]>.Run(() =>
                    {
                        double[] result = new double[input_values.Length];

                        for (int j = 0; j < BatchSize; ++j)
                        {
                            current.FeedForward(input_values, out result);
                        }

                        return result;
                    }, tokens[g]);
                }

                await Task.WhenAll(tasks);

                for (int k = 0; k < all_results.Length; ++k)
                {
                    all_results[k] = new double[input_values.Length];
                }

                for (int g = 0; g < GenerationSize; ++g)
                {
                    all_results[g] = tasks[g].Result;
                }

                Heuristics = new float[GenerationSize];
                for (int g = 0; g < GenerationSize; ++g)
                {
                    Heuristics[g] = ComputeHeuristic(all_results[g], test_values);
                }

                int best_index = -1;
                float best_heuristic = float.MaxValue;
                for (int h = 0; h < Heuristics.Length; ++h)
                {
                    if (Heuristics[h] < best_heuristic)
                    {
                        best_index = h;
                        best_heuristic = Heuristics[h];
                    }
                }

                //Debug.LogError($"Best Network Instance result is index {best_index} with heurisit {best_heuristic}.");
                double best_accuracy = 0;
                for (int g = 0; g < GenerationSize; ++g)
                {
                    bool result = CheckResult(Heuristics[g]);

                    AccuraciesData[g, 0] = result == true ? AccuraciesData[g, 0] + 1 : AccuraciesData[g, 0];
                    AccuraciesData[g, 1] = result == false ? AccuraciesData[g, 1] + 1 : AccuraciesData[g, 1];

                    float accuracy = ((float)AccuraciesData[g, 0] * 1) / (float)(AccuraciesData[g, 0] + AccuraciesData[g, 1]);
                    accuracy *= 100f;
                    AccuraciesData[g, 2] = accuracy;
                    Accuracies[g] = accuracy;

                    best_accuracy = Math.Max(best_accuracy, accuracy);
                }

                if (best_accuracy < Training_Best_Mean_Error)
                {
                    LowerBestAccuracyCount++;
                }

                if (best_accuracy > Training_Best_Mean_Error)
                {
                    // Keeping a trace of the set
                    MemorizeBestSet(currentNetworks[best_index], best_accuracy);
                    LowerBestAccuracyCount = 0;
                    ResetToBestCount = 0;
                }

                if (best_accuracy >= Training_Best_Mean_Error || LowerBestAccuracyCount < LowerBestAccuracyThreshold)
                {
                    for (int g = 0; g < GenerationSize; ++g)
                    {
                        // The best network will mutate independently
                        if (best_index != g)
                        {
                            ComputeChildrenWeights(currentNetworks[best_index], currentNetworks[g]);
                        }
                    }

                    Mutate(currentNetworks[best_index]);
                    //ResetToBestCount = 0;
                }
                else
                {
                    if (ResetToBestCount > BestSetAutoStop)
                    {
                        Debug.LogError("Training should stop or parameters should change");
                        LearningRate /= 2f;
                        //return;
                        LowerBestAccuracyCount = 0;
                        ResetToBestCount = 0;
                    }

                    //Debug.Log("Accuracy is lower than previous iteration");

                    // Reseting first network to the previous best set
                    SetWeightsFromBest(currentNetworks[0]);

                    // Generating childrens
                    for (int g = 1; g < GenerationSize; ++g)
                    {
                        ComputeChildrenWeights(currentNetworks[0], currentNetworks[g]);
                    }

                    ResetToBestCount++;
                }
            }

            SaveBestTrainingWeightSet(currentNetworks[0]);
        }

        public virtual float ComputeHeuristic(double[] result, double[] testValue)
        {
            float heuristic = 0;

            for (int i = 0; i < result.Length; ++i)
            {
                heuristic += Mathf.Abs((float)testValue[i] - (float)result[i]);
            }
            return heuristic / result.Length;
        }

        public bool CheckResult(float heuristic)
        {
            return heuristic < HeuristicThreshold;
        }

        public void ComputeChildrenWeights(NeuralNetwork parent, NeuralNetwork children)
        {
            for (int l = 0; l < parent.layers.Count; ++l)
            {
                for (int i = 0; i < parent.layers[l].Weights.GetLength(0); ++i)
                {
                    for (int k = 0; k < parent.layers[l].Weights.GetLength(1); ++k)
                    {
                        // The delta is a random mutation applied on each weight
                        double delta = UnityEngine.Random.Range(-LearningRate, LearningRate);

                        // The children weight is the parent weight + the delta
                        children.layers[l].Weights[i, k] = delta + parent.layers[l].Weights[i, k];

                        // We store the delta vector on each children
                        // If it appears to be the best parent of the next epoch
                        // The network will use this delta vector as a momentum and keep the direction of changes
                        children.layers[l].PreviousWeightDelta[i, k] = delta;
                        // We add the delta vector of the parent * Momentum ratio
                        children.layers[l].Weights[i, k] += Momentum * parent.layers[l].PreviousWeightDelta[i, k];
                        // Decaying the weight can be useful to avoid overlearning
                        children.layers[l].Weights[i, k] -= WeightDecay * children.layers[l].Weights[i, k];
                    }
                }

                for (int i = 0; i < parent.layers[l].Biases.Length; ++i)
                {
                    double biaseDelta = UnityEngine.Random.Range(-LearningRate, LearningRate) * BiasRate;

                    children.layers[l].Biases[i] = biaseDelta + parent.layers[l].Biases[i];
                    children.layers[l].PreviousBiasesDelta[i] = biaseDelta;
                    children.layers[l].Biases[i] += Momentum * parent.layers[l].PreviousBiasesDelta[i];
                    children.layers[l].Biases[i] -= WeightDecay * children.layers[l].Biases[i];
                }
            }
        }

        public void Mutate(NeuralNetwork neuralNetwork)
        {
            for (int l = 0; l < neuralNetwork.layers.Count; ++l)
            {
                for (int i = 0; i < neuralNetwork.layers[l].Weights.GetLength(0); ++i)
                {
                    for (int k = 0; k < neuralNetwork.layers[l].Weights.GetLength(1); ++k)
                    {
                        if (UnityEngine.Random.Range(0f, 100f) > 100f - MutationChancesPurcent)
                        {
                            double delta = UnityEngine.Random.Range(-MutationRate, MutationRate);

                            neuralNetwork.layers[l].Weights[i, k] = delta + neuralNetwork.layers[l].Weights[i, k];
                            neuralNetwork.layers[l].PreviousWeightDelta[i, k] = delta;
                            neuralNetwork.layers[l].Weights[i, k] -= WeightDecay * neuralNetwork.layers[l].Weights[i, k];
                        }
                        else
                        {
                            neuralNetwork.layers[l].Weights[i, k] += Momentum * neuralNetwork.layers[l].PreviousWeightDelta[i, k];
                            neuralNetwork.layers[l].Weights[i, k] -= WeightDecay * neuralNetwork.layers[l].Weights[i, k];
                        }
                    }
                }

                for (int i = 0; i < neuralNetwork.layers[l].Biases.Length; ++i)
                {
                    if (UnityEngine.Random.Range(0f, 100f) > 100f - MutationChancesPurcent)
                    {
                        double biaseDelta = UnityEngine.Random.Range(-MutationRate, MutationRate) * BiasRate;

                        neuralNetwork.layers[l].Biases[i] = biaseDelta + neuralNetwork.layers[l].Biases[i];
                        neuralNetwork.layers[l].PreviousBiasesDelta[i] = biaseDelta;
                        neuralNetwork.layers[l].Biases[i] -= WeightDecay * neuralNetwork.layers[l].Biases[i];
                    }
                    else
                    {
                        neuralNetwork.layers[l].Biases[i] += Momentum * neuralNetwork.layers[l].PreviousBiasesDelta[i];
                        neuralNetwork.layers[l].Biases[i] -= WeightDecay * neuralNetwork.layers[l].Biases[i];
                    }
                }
            }
        }
    }
}

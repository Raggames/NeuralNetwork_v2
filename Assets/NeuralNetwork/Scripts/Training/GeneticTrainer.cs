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

        public NetworkBuilder Builder;
        public List<NeuralNetwork> currentNetworks = new List<NeuralNetwork>();

        [Header("---- PARAMETERS -----")]
        ///The maximum of iterations without finding any better set than Epoch_Best_Weigths
        public int BestSetAutoStop = 1000;
        public int LowerBestAccuracyThreshold = 500;

        public int GenerationSize = 10;

        public float MutationRate = 0.001f;
        public float MutationChancesPurcent = 3;

        [Header("----RUNTIME----")]

        public double[,] Accuracies;
        public double[] AccuracyDebug;

        public double[][,] Training_Best_Weigths;
        public double[][] Training_Best_Biases;
        public double Training_Best_Accuracy;

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

        public async void Train()
        {
            Training_Best_Accuracy = -1;
            TrainingSetting.Init();

            // Each epoch, we keep a trace of the best set
            // If the next one doesn't find a best set, we will retry from this one
            // If the retry iterations are over a threshold, the learning will stop and keed this set as the best
            Training_Best_Weigths = new double[currentNetworks[0].layers.Count][,];
            Training_Best_Biases = new double[currentNetworks[0].layers.Count][];

            for (int i = 0; i < currentNetworks[0].layers.Count; ++i)
            {
                Training_Best_Weigths[i] = NeuralNetworkMathHelper.MakeMatrix(currentNetworks[0].layers[i].Weights.GetLength(0), currentNetworks[0].layers[i].Weights.GetLength(1));
                Training_Best_Biases[i] = new double[currentNetworks[0].layers[i].Biases.Length];
            }

            Accuracies = new double[GenerationSize, 3];
            AccuracyDebug = new double[GenerationSize];

            for (int i = 0; i < Epochs; ++i)
            {
                CurrentEpoch = i;

                double[] input_values;
                double[] test_values;
                TrainingSetting.GetNextValues(out input_values, out test_values);

                Task<double[]>[] tasks = new Task<double[]>[GenerationSize];
                CancellationToken[] tokens = new CancellationToken[GenerationSize];

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

                double[][] all_results = new double[GenerationSize][];
                for (int k = 0; k < all_results.Length; ++k)
                {
                    all_results[k] = new double[input_values.Length];
                }

                for (int g = 0; g < GenerationSize; ++g)
                {
                    all_results[g] = tasks[g].Result;
                }

                float[] heuristics = new float[GenerationSize];
                for (int g = 0; g < GenerationSize; ++g)
                {
                    heuristics[g] = ComputeHeuristic(all_results[g], test_values);
                }

                int best_index = -1;
                float best_heuristic = float.MaxValue;
                for (int h = 0; h < heuristics.Length; ++h)
                {
                    if (heuristics[h] < best_heuristic)
                    {
                        best_index = h;
                        best_heuristic = heuristics[h];
                    }
                }

                //Debug.LogError($"Best Network Instance result is index {best_index} with heurisit {best_heuristic}.");
                double best_accuracy = 0;
                for (int g = 0; g < GenerationSize; ++g)
                {
                    bool result = CheckResult(heuristics[g]);

                    Accuracies[g, 0] = result == true ? Accuracies[g, 0] + 1 : Accuracies[g, 0];
                    Accuracies[g, 1] = result == false ? Accuracies[g, 1] + 1 : Accuracies[g, 1];

                    float accuracy = ((float)Accuracies[g, 0] * 1) / (float)(Accuracies[g, 0] + Accuracies[g, 1]);
                    accuracy *= 100f;
                    Accuracies[g, 2] = accuracy;
                    AccuracyDebug[g] = accuracy;

                    best_accuracy = Math.Max(best_accuracy, accuracy);
                }

                if (best_accuracy < Training_Best_Accuracy)
                {
                    LowerBestAccuracyCount++;
                }

                if (best_accuracy > Training_Best_Accuracy)
                {
                    // Keeping a trace of the set
                    MemorizeBestSet(currentNetworks[best_index], best_accuracy);
                    LowerBestAccuracyCount = 0;
                    ResetToBestCount = 0;
                }

                if (best_accuracy >= Training_Best_Accuracy || LowerBestAccuracyCount < LowerBestAccuracyThreshold)
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
        }

        public float ComputeHeuristic(double[] result, double[] testValue)
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
            return heuristic < .01f;
        }

        private void SetWeightsFromBest(NeuralNetwork network)
        {
            for (int l = 0; l < network.layers.Count; ++l)
            {
                for (int i = 0; i < network.layers[l].Weights.GetLength(0); ++i)
                {
                    for (int k = 0; k < network.layers[l].Weights.GetLength(1); ++k)
                    {
                        network.layers[l].Weights[i, k] = Training_Best_Weigths[l][i, k];
                    }
                }

                for (int i = 0; i < network.layers[l].Biases.Length; ++i)
                {
                    network.layers[l].Biases[i] = Training_Best_Biases[l][i];
                }
            }
        }

        private void MemorizeBestSet(NeuralNetwork bestSet, double accuracy)
        {
            Training_Best_Accuracy = accuracy;

            for (int l = 0; l < bestSet.layers.Count; ++l)
            {
                for (int i = 0; i < bestSet.layers[l].Weights.GetLength(0); ++i)
                {
                    for (int k = 0; k < bestSet.layers[l].Weights.GetLength(1); ++k)
                    {
                        Training_Best_Weigths[l][i, k] = bestSet.layers[l].Weights[i, k];
                    }
                }

                for (int i = 0; i < bestSet.layers[l].Biases.Length; ++i)
                {
                    Training_Best_Biases[l][i] = bestSet.layers[l].Biases[i];
                }
            }
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

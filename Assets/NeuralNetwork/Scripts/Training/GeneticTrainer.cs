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

        public int GenerationSize = 10;

        [Header("----RUNTIME----")]

        public double[][] TestingDataX;
        public double[][] TestingDataY;
        public double[,] Accuracies;
        public double[] AccuracyDebug;

        private void Start()
        {
            CreateNetworks();
            Train();
        }

        private void CreateNetworks()
        {
            currentNetworks.Clear();

            for(int i = 0; i < GenerationSize; ++i)
            {
                currentNetworks.Add(new NeuralNetwork());
            }

            for (int i = 0; i < GenerationSize; ++i)
            {
                currentNetworks[i].CreateNetwork(this, Builder);
            }
        }

        private void InitTraining()
        {
            TestingDataX = new double[15000][];
            TestingDataY = new double[15000][];

            for (int i = 0; i < TestingDataX.Length; ++i)
            {
                TestingDataX[i] = new double[4];
                TestingDataY[i] = new double[1];

                //double[] data = DataManager.GetDataEntry(i);

                TestingDataX[i][0] = UnityEngine.Random.Range(-1f, 1f);
                TestingDataX[i][1] = UnityEngine.Random.Range(-1f, 1f);
                TestingDataX[i][2] = UnityEngine.Random.Range(-1f, 1f);
                TestingDataX[i][3] = UnityEngine.Random.Range(-1f, 1f);

                double sum = TestingDataX[i][0] + TestingDataX[i][1] + TestingDataX[i][2] + TestingDataX[i][3];
                TestingDataY[i][0] = sum > 0 ? 1 : 0;
            }
        }

        public async void Train()
        {
            InitTraining();

            Accuracies = new double[GenerationSize, 3];
            AccuracyDebug = new double[GenerationSize];

            for (int i = 0; i < Epochs; ++i)
            {
                CurrentEpoch = i;

                int index = UnityEngine.Random.Range(0, TestingDataX.Length);
                double[] input_values = TestingDataX[index];
                double[] test_values = TestingDataY[index];

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
                            current.ComputeFeedForward(input_values, out result);
                        }

                        return result;
                    }, tokens[g]);
                }
                
                await Task.WhenAll(tasks);

                double[][] all_results = new double[GenerationSize][];
                for(int k = 0; k < all_results.Length; ++k)
                {
                    all_results[k] = new double[input_values.Length];
                }

                for (int g = 0; g < GenerationSize; ++g)
                {
                    all_results[g] = tasks[g].Result;
                }

                float[] heuristics = new float[GenerationSize];
                for(int g = 0; g < GenerationSize; ++g)
                {
                    heuristics[g] = ComputeHeuristic(all_results[g], test_values);
                }

                int best_index = -1;
                float best_heuristic = float.MaxValue;
                for(int h = 0; h < heuristics.Length; ++h)
                {
                    if(heuristics[h] < best_heuristic)
                    {
                        best_index = h;
                        best_heuristic = heuristics[h];
                    }
                }

                //Debug.LogError($"Best Network Instance result is index {best_index} with heurisit {best_heuristic}.");

                double[] best_weights = currentNetworks[best_index].GetWeights();
                for (int g = 0; g < GenerationSize; ++g)
                {
                    bool result = CheckResult(heuristics[g]);
                    Accuracies[g, 0] = result == true ? Accuracies[g, 0] + 1 : Accuracies[g, 0];
                    Accuracies[g, 1] = result == false ? Accuracies[g, 1] + 1 : Accuracies[g, 1];

                    float accuracy = ((float)Accuracies[g, 0] * 1) / (float)(Accuracies[g, 0] + Accuracies[g, 1]);
                    accuracy *= 100f;
                    Accuracies[g, 2] = accuracy;
                    AccuracyDebug[g] = accuracy;

                    currentNetworks[g].SetWeights(GenerateWeigth(best_weights));
                }
            }
        }

        public float ComputeHeuristic(double[] result, double[] testValue)
        {
            float heuristic = 0;

            for(int i = 0; i < result.Length; ++i)
            {
                heuristic += Mathf.Abs((float)testValue[i] - (float)result[i]);
            }
            return heuristic / result.Length;
        }

        public bool CheckResult(float heuristic)
        {
            return heuristic < .01f;
        }

        public double[] GenerateWeigth(double[] input)
        {
            for(int i = 0; i < input.Length; ++i)
            {
                float delta = UnityEngine.Random.Range(-LearningRate, LearningRate);
                input[i] += delta; 
            }

            return input;
        }
    }
}

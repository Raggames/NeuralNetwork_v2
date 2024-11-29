using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core
{
    public class SGDUnsupervisedModelTuningSystem<KModelTuningProfile, TTrainer, TModel, TModelInput, TModelOutput> : HyperparameterTuningSystemBase<KModelTuningProfile, IStochasticGradientDescentParameters, TTrainer, TModel, TModelInput, TModelOutput>
            where KModelTuningProfile : ITuningProfile<IStochasticGradientDescentParameters>
            where TModel : IMLModel<TModelInput, TModelOutput>
            where TTrainer : IMLTrainer<TModel, TModelInput, TModelOutput>
            where TModelInput : ICloneable
    {
        private object _lock = new object();

        public struct HyperparameterData : IStochasticGradientDescentParameters
        {
            public double Score { get; set; }

            public float Epochs { get; set; }
            public float BatchSize { get; set; }
            public float LearningRate { get; set; }
            public float BiasRate { get; set; }
            public float Momentum { get; set; }
            public float WeightDecay { get; set; }

            public HyperparameterData(double score, IStochasticGradientDescentParameters parameter)
            {
                Score = score;
                Epochs = parameter.Epochs;
                BatchSize = parameter.BatchSize;
                LearningRate = parameter.LearningRate;
                BiasRate = parameter.BiasRate;
                Momentum = parameter.Momentum;
                WeightDecay = parameter.WeightDecay;
            }
        }

        public override async Task<IStochasticGradientDescentParameters> Search(int iterations, KModelTuningProfile kModelTuningProfile, TModelInput[] t_inputs, TTrainer[] trainers)
        {
            //await Task.Delay(1);
            var learning_rate = 1;

            double[][] scores = new double[iterations][];
            HyperparameterData[][] hyperparametersHistory = new HyperparameterData[iterations][];
            var besthyperparameterDatas = new List<HyperparameterData>();

            /// cloning datas to avoid race condition while parallel processing on the same dataset
            TModelInput[][] datas = new TModelInput[trainers.Length][];
            for (int i = 0; i < trainers.Length; ++i)
            {
                datas[i] = (TModelInput[])t_inputs.Clone();
            }

            // random init of the hyperparameter 'vector'
            foreach (var trainer in trainers)
            {
                var trainerParam = (trainer as IStochasticGradientDescentParameters);

                trainerParam.Epochs = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Epochs, kModelTuningProfile.UpperBound.Epochs);
                trainerParam.BatchSize = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BatchSize, kModelTuningProfile.UpperBound.BatchSize);
                trainerParam.LearningRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.LearningRate, kModelTuningProfile.UpperBound.LearningRate);
                trainerParam.BiasRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BiasRate, kModelTuningProfile.UpperBound.BiasRate);
                trainerParam.Momentum = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Momentum, kModelTuningProfile.UpperBound.Momentum);
                trainerParam.WeightDecay = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.WeightDecay, kModelTuningProfile.UpperBound.WeightDecay);
            }

            int it_index = 0;
            while (it_index < iterations)
            {
                scores[it_index] = new double[trainers.Length];
                hyperparametersHistory[it_index] = new HyperparameterData[trainers.Length];

                Debug.Log($"Tuner enter iteration {it_index}");

                var fitTasks = trainers.Select(async (trainer, index) =>
                {
                    await Task.Delay(1);

                    /*// Fit the trainer asynchronously
                    var tr_result = trainer.FitSynchronously(t_inputs);

                    // Score the trainer asynchronously
                    var tr_score = trainer.ScoreSynchronously();*/

                    // Fit the trainer asynchronously
                    var tr_result = await trainer.Fit(datas[index]);

                    // Score the trainer asynchronously
                    var tr_score = await trainer.Score();

                    // Safely update the scores array
                    lock (_lock)
                    {
                        hyperparametersHistory[it_index][index] = new HyperparameterData(tr_score, trainer as IStochasticGradientDescentParameters);
                        scores[it_index][index] = tr_score;
                    }
                }).ToArray();


                // Wait for all tasks to complete
                await Task.WhenAll(fitTasks);

                //Debug.Log($"Tuner end fit iteration {it_index}");

                var iteration_best_score = double.MinValue;
                int iteration_best_score_index = -1;
                var iteration_lowest_score = double.MaxValue;
                int iteration_lowest_score_index = -1;

                for (int i = 0; i < scores[it_index].Length; ++i)
                {
                    Debug.Log($"Tuner {i} scored {scores[it_index][i]}");

                    if (scores[it_index][i] > iteration_best_score)
                    {
                        iteration_best_score = scores[it_index][i];
                        iteration_best_score_index = i;
                    }
                    else if (scores[it_index][i] < iteration_lowest_score)
                    {
                        iteration_lowest_score = scores[it_index][i];
                        iteration_lowest_score_index = i;
                    }
                }

                besthyperparameterDatas.Add(new HyperparameterData(iteration_best_score, trainers[iteration_best_score_index] as IStochasticGradientDescentParameters));

                //UpdateWithGradients(kModelTuningProfile, trainers, learning_rate, hyperparametersHistory, besthyperparameterDatas, it_index, iteration_best_score_index, iteration_lowest_score_index);
                UpdateGenetic(kModelTuningProfile, trainers, learning_rate, hyperparametersHistory, besthyperparameterDatas, it_index, iteration_best_score_index, iteration_lowest_score_index);

                it_index++;
            }


            var best_overall_score = double.MinValue;
            int best_overall_score_index = -1;

            for (int i = 0; i < besthyperparameterDatas.Count; ++i)
            {
                Debug.Log($"Tuner {i} hp score > {besthyperparameterDatas[i].Score}");

                if (besthyperparameterDatas[i].Score > best_overall_score)
                {
                    best_overall_score = besthyperparameterDatas[i].Score;
                    best_overall_score_index = i;
                }
            }

            Debug.Log($"Best overall hp score > {besthyperparameterDatas[best_overall_score_index].Score}");

            //return bestHyperparameterDatas[best_overall_score_index];

            Debug.Log($"End fit. Best params : " +
              $"Epochs = {besthyperparameterDatas[best_overall_score_index].Epochs}, " +
              $"Batchsize = {besthyperparameterDatas[best_overall_score_index].BatchSize}, " +
              $"LearningRate = {besthyperparameterDatas[best_overall_score_index].LearningRate}, " +
              $"BiasRate = {besthyperparameterDatas[best_overall_score_index].BiasRate}, " +
              $"Momentum = {besthyperparameterDatas[best_overall_score_index].Momentum}, " +
              $"WeightDecay = {besthyperparameterDatas[best_overall_score_index].WeightDecay}, ");

            return besthyperparameterDatas[best_overall_score_index];
        }

        private void UpdateWithGradients(KModelTuningProfile kModelTuningProfile, TTrainer[] trainers, float learning_rate, HyperparameterData[][] hyperparametersHistory, List<HyperparameterData> besthyperparameterDatas, int it_index, int iteration_best_score_index, int iteration_lowest_score_index)
        {
            int trainer_index = 0;

            // init random values in the search space given by the profile
            foreach (var trainer in trainers)
            {
                var trainerParam = (trainer as IStochasticGradientDescentParameters);

                if (it_index < 2)
                {
                    var best_iteration_vector = (trainers[iteration_best_score_index] as IStochasticGradientDescentParameters).GetHyperparameterVector();
                    var low_iteration_vector = (trainers[iteration_lowest_score_index] as IStochasticGradientDescentParameters).GetHyperparameterVector();
                    var gradient_vector = (best_iteration_vector - low_iteration_vector);

                    Debug.Log($"Hyperparameter diff at {it_index}th iteration > {gradient_vector}");

                    trainerParam.Epochs = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Epochs, kModelTuningProfile.UpperBound.Epochs);
                    trainerParam.BatchSize = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BatchSize, kModelTuningProfile.UpperBound.BatchSize);
                    trainerParam.LearningRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.LearningRate, kModelTuningProfile.UpperBound.LearningRate);
                    trainerParam.BiasRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BiasRate, kModelTuningProfile.UpperBound.BiasRate);
                    trainerParam.Momentum = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Momentum, kModelTuningProfile.UpperBound.Momentum);
                    trainerParam.WeightDecay = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.WeightDecay, kModelTuningProfile.UpperBound.WeightDecay);
                }
                else
                {
                    var sorted_hp = besthyperparameterDatas.OrderByDescending(t => t.Score).ToArray();
                    var best_tparam = (sorted_hp[0] as IStochasticGradientDescentParameters);
                    var best_overall_hp_vector = best_tparam.GetHyperparameterVector();

                    var local_hp_vector = trainerParam.GetHyperparameterVector();
                    NVector gradient_vector = new NVector(6);

                    // half of trainers will do gradient descent
                    /*if (trainer_index % 2 == 1)
                    {
                        var previous_hp = hyperparametersHistory[it_index - 1][trainer_index];
                        var previous_score = scores[it_index - 1][trainer_index];

                        gradient_vector = (local_hp_vector - (hyperparametersHistory[it_index - 2][trainer_index] as IStochasticGradientDescentParameters).GetHyperparameterVector());
                        if (previous_score > scores[it_index][trainer_index])
                        {
                            gradient_vector *= -1f;
                        }

                        Debug.Log($"Hyperparameter gradient for highest at {it_index}th iteration > {gradient_vector}");

                        if (gradient_vector.magnitude < .001)
                        {
                            for (int i = 0; i < gradient_vector.length; i++)
                            {
                                gradient_vector[i] = MLRandom.Shared.Range(-1.0, 1.0) * learning_rate;
                            }
                        }

                        trainerParam.Epochs += (int)Math.Round((kModelTuningProfile.UpperBound.Epochs - kModelTuningProfile.UpperBound.Epochs) * learning_rate * gradient_vector[0]);
                        trainerParam.BatchSize += (int)Math.Round((kModelTuningProfile.UpperBound.BatchSize - kModelTuningProfile.UpperBound.BatchSize) * learning_rate * gradient_vector[1]);

                        trainerParam.LearningRate += (float)((kModelTuningProfile.UpperBound.LearningRate - kModelTuningProfile.UpperBound.LearningRate) * learning_rate * gradient_vector[2]);
                        trainerParam.BiasRate += (float)((kModelTuningProfile.UpperBound.BiasRate - kModelTuningProfile.UpperBound.BiasRate) * learning_rate * gradient_vector[3]);
                        trainerParam.Momentum += (float)((kModelTuningProfile.UpperBound.Momentum - kModelTuningProfile.UpperBound.Momentum) * learning_rate * gradient_vector[4]);
                        trainerParam.WeightDecay += (float)((kModelTuningProfile.UpperBound.WeightDecay - kModelTuningProfile.UpperBound.WeightDecay) * learning_rate * gradient_vector[5]);
                    }*/
                    if (trainer_index % 2 == 1)
                    {
                        var previous_hp = hyperparametersHistory[it_index - 1][trainer_index];

                        int v = 1;
                        while (gradient_vector.magnitude < .0001 && v < sorted_hp.Length)
                        {
                            var vector_under = (sorted_hp[v] as IStochasticGradientDescentParameters).GetHyperparameterVector();
                            gradient_vector = (local_hp_vector - vector_under);

                            v++;
                        }


                        Debug.Log($"GD for {it_index}th iteration > {gradient_vector}");

                        if (gradient_vector.magnitude < .001)
                        {
                            trainerParam.Epochs = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Epochs, kModelTuningProfile.UpperBound.Epochs);
                            trainerParam.BatchSize = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BatchSize, kModelTuningProfile.UpperBound.BatchSize);
                            trainerParam.LearningRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.LearningRate, kModelTuningProfile.UpperBound.LearningRate);
                            trainerParam.BiasRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BiasRate, kModelTuningProfile.UpperBound.BiasRate);
                            trainerParam.Momentum = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Momentum, kModelTuningProfile.UpperBound.Momentum);
                            trainerParam.WeightDecay = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.WeightDecay, kModelTuningProfile.UpperBound.WeightDecay);
                        }
                        else
                        {
                            trainerParam.Epochs += (int)Math.Round((kModelTuningProfile.UpperBound.Epochs - kModelTuningProfile.LowerBound.Epochs) * learning_rate * gradient_vector[0]);
                            trainerParam.BatchSize += (int)Math.Round((kModelTuningProfile.UpperBound.BatchSize - kModelTuningProfile.LowerBound.BatchSize) * learning_rate * gradient_vector[1]);

                            trainerParam.LearningRate += (float)((kModelTuningProfile.UpperBound.LearningRate - kModelTuningProfile.LowerBound.LearningRate) * learning_rate * gradient_vector[2]);
                            trainerParam.BiasRate += (float)((kModelTuningProfile.UpperBound.BiasRate - kModelTuningProfile.LowerBound.BiasRate) * learning_rate * gradient_vector[3]);
                            trainerParam.Momentum += (float)((kModelTuningProfile.UpperBound.Momentum - kModelTuningProfile.LowerBound.Momentum) * learning_rate * gradient_vector[4]);
                            trainerParam.WeightDecay += (float)((kModelTuningProfile.UpperBound.WeightDecay - kModelTuningProfile.LowerBound.WeightDecay) * learning_rate * gradient_vector[5]);

                        }
                    }
                    // other half will explore with more randomness
                    // as all gradient are shared, they can use gradient descending hp parameter if they are good to align
                    else
                    {
                        if (trainer_index % 2 == 0 && MLRandom.Shared.Range(0.0, 1.0) > .8)
                        {
                            trainerParam.Epochs = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Epochs, kModelTuningProfile.UpperBound.Epochs);
                            trainerParam.BatchSize = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BatchSize, kModelTuningProfile.UpperBound.BatchSize);
                            trainerParam.LearningRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.LearningRate, kModelTuningProfile.UpperBound.LearningRate);
                            trainerParam.BiasRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BiasRate, kModelTuningProfile.UpperBound.BiasRate);
                            trainerParam.Momentum = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Momentum, kModelTuningProfile.UpperBound.Momentum);
                            trainerParam.WeightDecay = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.WeightDecay, kModelTuningProfile.UpperBound.WeightDecay);
                        }
                        else
                        {
                            gradient_vector = (best_overall_hp_vector - local_hp_vector);
                            Debug.Log($"RD at {it_index}th iteration > {gradient_vector}");

                            if (gradient_vector.magnitude < .001)
                            {
                                trainerParam.Epochs = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Epochs, kModelTuningProfile.UpperBound.Epochs);
                                trainerParam.BatchSize = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BatchSize, kModelTuningProfile.UpperBound.BatchSize);
                                trainerParam.LearningRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.LearningRate, kModelTuningProfile.UpperBound.LearningRate);
                                trainerParam.BiasRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BiasRate, kModelTuningProfile.UpperBound.BiasRate);
                                trainerParam.Momentum = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Momentum, kModelTuningProfile.UpperBound.Momentum);
                                trainerParam.WeightDecay = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.WeightDecay, kModelTuningProfile.UpperBound.WeightDecay);
                            }
                            else
                            {
                                trainerParam.Epochs += (int)Math.Round((kModelTuningProfile.UpperBound.Epochs - kModelTuningProfile.LowerBound.Epochs) * learning_rate * gradient_vector[0]);
                                trainerParam.BatchSize += (int)Math.Round((kModelTuningProfile.UpperBound.BatchSize - kModelTuningProfile.LowerBound.BatchSize) * learning_rate * gradient_vector[1]);

                                trainerParam.LearningRate += (float)((kModelTuningProfile.UpperBound.LearningRate - kModelTuningProfile.LowerBound.LearningRate) * learning_rate * gradient_vector[2]);
                                trainerParam.BiasRate += (float)((kModelTuningProfile.UpperBound.BiasRate - kModelTuningProfile.LowerBound.BiasRate) * learning_rate * gradient_vector[3]);
                                trainerParam.Momentum += (float)((kModelTuningProfile.UpperBound.Momentum - kModelTuningProfile.LowerBound.Momentum) * learning_rate * gradient_vector[4]);
                                trainerParam.WeightDecay += (float)((kModelTuningProfile.UpperBound.WeightDecay - kModelTuningProfile.LowerBound.WeightDecay) * learning_rate * gradient_vector[5]);
                            }
                        }
                    }
                }

                trainer_index++;
            }
        }

        NVector _previous_best_hpvec = new NVector(6);

        private void UpdateGenetic(KModelTuningProfile kModelTuningProfile, TTrainer[] trainers, float learning_rate, HyperparameterData[][] hyperparametersHistory, List<HyperparameterData> besthyperparameterDatas, int it_index, int iteration_best_score_index, int iteration_lowest_score_index)
        {
            // mutation rate is a chance of having 
            float mutation_rate = .25f;
            learning_rate = .05f;

            int trainer_index = 0;

            // init random values in the search space given by the profile
            foreach (var trainer in trainers)
            {
                var trainerParam = (trainer as IStochasticGradientDescentParameters);
                var sorted_hp = besthyperparameterDatas.OrderByDescending(t => t.Score).ToArray();
                var current_best_hpvec = (sorted_hp[0] as IStochasticGradientDescentParameters).GetHyperparameterVector();

                if (it_index < 2)
                {
                    trainerParam.Epochs = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Epochs, kModelTuningProfile.UpperBound.Epochs);
                    trainerParam.BatchSize = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BatchSize, kModelTuningProfile.UpperBound.BatchSize);
                    trainerParam.LearningRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.LearningRate, kModelTuningProfile.UpperBound.LearningRate);
                    trainerParam.BiasRate = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.BiasRate, kModelTuningProfile.UpperBound.BiasRate);
                    trainerParam.Momentum = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.Momentum, kModelTuningProfile.UpperBound.Momentum);
                    trainerParam.WeightDecay = MLRandom.Shared.Range(kModelTuningProfile.LowerBound.WeightDecay, kModelTuningProfile.UpperBound.WeightDecay);

                    continue;
                }

                var local_vec = trainerParam.GetHyperparameterVector();

                // if the best hpvec has changed, we use the best as new origin for a part of the trainers 
                if (current_best_hpvec == _previous_best_hpvec)
                {
                    trainerParam.Epochs = (int)Math.Round(current_best_hpvec[0]);
                    trainerParam.BatchSize = (int)Math.Round(current_best_hpvec[1]);
                    trainerParam.LearningRate = (float)current_best_hpvec[2];
                    trainerParam.BiasRate = (float)current_best_hpvec[3];
                    trainerParam.Momentum = (float)current_best_hpvec[4];
                    trainerParam.WeightDecay = (float)current_best_hpvec[5];
                }
                // else we continue mutating from the same
               
                // mutate
                trainerParam.Epochs += MLRandom.Shared.Range(0.0, 1.0) > 1.0 - mutation_rate ? (int)Math.Round((kModelTuningProfile.UpperBound.Epochs - kModelTuningProfile.LowerBound.Epochs) * learning_rate) : 0;
                trainerParam.BatchSize += MLRandom.Shared.Range(0.0, 1.0) > 1.0 - mutation_rate ? (int)Math.Round((kModelTuningProfile.UpperBound.BatchSize - kModelTuningProfile.LowerBound.BatchSize) * learning_rate) : 0;
                trainerParam.LearningRate += MLRandom.Shared.Range(0.0, 1.0) > 1.0 - mutation_rate ? (float)((kModelTuningProfile.UpperBound.LearningRate - kModelTuningProfile.LowerBound.LearningRate) * learning_rate) : 0;
                trainerParam.BiasRate += MLRandom.Shared.Range(0.0, 1.0) > 1.0 - mutation_rate ? (float)((kModelTuningProfile.UpperBound.BiasRate - kModelTuningProfile.LowerBound.BiasRate) * learning_rate) : 0;
                trainerParam.Momentum += MLRandom.Shared.Range(0.0, 1.0) > 1.0 - mutation_rate ? (float)((kModelTuningProfile.UpperBound.Momentum - kModelTuningProfile.LowerBound.Momentum) * learning_rate) : 0;
                trainerParam.WeightDecay += MLRandom.Shared.Range(0.0, 1.0) > 1.0 - mutation_rate ? (float)((kModelTuningProfile.UpperBound.WeightDecay - kModelTuningProfile.LowerBound.WeightDecay) * learning_rate) : 0;

                _previous_best_hpvec = current_best_hpvec;
                trainer_index++;
            }
        }
    }
}

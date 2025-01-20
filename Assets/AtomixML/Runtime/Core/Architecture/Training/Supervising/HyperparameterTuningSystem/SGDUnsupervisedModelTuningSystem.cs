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
    public enum TuningSystemOptimizationAlgorithms
    {
        Gradient,
        Genetic,
    }

    public class SGDUnsupervisedModelTuningSystem<KModelTuningProfile, TTrainer, TModel, TModelInput, TModelOutput> : HyperparameterTuningSystemBase<KModelTuningProfile, IStochasticGradientDescentParameters, TTrainer, TModel, TModelInput, TModelOutput>
            where KModelTuningProfile : ITuningProfile<IStochasticGradientDescentParameters>
            where TModel : IMLModel<TModelInput, TModelOutput>
            where TTrainer : IMLTrainer<TModel, TModelInput, TModelOutput>
            where TModelInput : ICloneable
    {
        private object _lock = new object();
        private int _iterations = 0;
        private float _learning_rate = 1f;

        public TuningSystemOptimizationAlgorithms optimizationAlgorithm { get; set; } = TuningSystemOptimizationAlgorithms.Gradient;

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
            _iterations = iterations;

            //await Task.Delay(1);
            var learning_rate = .25f;

            double[][] scores = new double[iterations][];
            HyperparameterData[][] hyperparametersHistory = new HyperparameterData[iterations][];
            var besthyperparameterDatas = new List<HyperparameterData>();

            /// cloning datas to avoid race condition while parallel processing on the same dataset
            TModelInput[][] datas = new TModelInput[trainers.Length][];
            for (int i = 0; i < trainers.Length; ++i)
            {
                datas[i] = (TModelInput[])t_inputs.Clone();
            }

            MLRandom.SeedShared((int)DateTime.Now.Ticks);
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

                switch (optimizationAlgorithm)
                {
                    case TuningSystemOptimizationAlgorithms.Gradient:
                        UpdateWithGradients(kModelTuningProfile, trainers, hyperparametersHistory, besthyperparameterDatas, it_index, iteration_best_score_index, iteration_lowest_score_index);
                        break;
                    case TuningSystemOptimizationAlgorithms.Genetic:
                        UpdateGenetic(kModelTuningProfile, trainers, learning_rate, hyperparametersHistory, besthyperparameterDatas, it_index, iteration_best_score_index, iteration_lowest_score_index);
                        break;
                }

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

        NVector _total_gradient_vector_values = new NVector(6);
        NVector _total_gradient_vector_sum = new NVector(6);

        private void UpdateWithGradients(KModelTuningProfile kModelTuningProfile, TTrainer[] trainers, HyperparameterData[][] hyperparametersHistory, List<HyperparameterData> besthyperparameterDatas, int it_index, int iteration_best_score_index, int iteration_lowest_score_index)
        {
            var current_learning_rate = (1f - ((float)it_index / (float)_iterations)) * _learning_rate;

            int trainer_index = 0;
            float gradient_rate = 1;

            // init random values in the search space given by the profile
            foreach (var trainer in trainers)
            {
                var trainerParam = (trainer as IStochasticGradientDescentParameters);

                if (it_index < 1)
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
                    var previous_hp = (hyperparametersHistory[it_index - 1][trainer_index] as IStochasticGradientDescentParameters).GetHyperparameterVector();
                    var current_hp = (hyperparametersHistory[it_index][trainer_index] as IStochasticGradientDescentParameters).GetHyperparameterVector();

                    var vect_diff = current_hp - previous_hp;
                    var score_diff = hyperparametersHistory[it_index][trainer_index].Score - hyperparametersHistory[it_index - 1][trainer_index].Score;

                    _total_gradient_vector_values += vect_diff * score_diff;
                    _total_gradient_vector_sum += vect_diff.Absolute();
                }

                trainer_index++;
            }

            if (it_index < 1)
                return;

            // gradient is the average of the previous runs - runs -1 hp vector difference ponderated by the score change
            // if the score between current and previous iteration is better, we add the difference, else we add the negative difference
            var current_gradient = new NVector(6);
            for(int i = 0; i < current_gradient.length; i++)
            {
                current_gradient[i] = _total_gradient_vector_values[i] / _total_gradient_vector_sum[i];
            }

            foreach (var trainer in trainers)
            {
                var trainerParam = (trainer as IStochasticGradientDescentParameters);

                Debug.Log(trainerParam.GetHashCode());

                // each feature will be a random part (-1, 1) * learningRate * (max-min) + gradient * gradient_rate
                trainerParam.Epochs += (float)MLRandom.Shared.Range(-1.0, 1.0) * (kModelTuningProfile.UpperBound.Epochs - kModelTuningProfile.LowerBound.Epochs) * current_learning_rate + (float)current_gradient[0] * gradient_rate;
                trainerParam.BatchSize += (float)MLRandom.Shared.Range(-1.0, 1.0)  * (kModelTuningProfile.UpperBound.BatchSize - kModelTuningProfile.LowerBound.BatchSize) * current_learning_rate + (float)current_gradient[1] * gradient_rate;
                trainerParam.LearningRate += (float)MLRandom.Shared.Range(-1.0, 1.0) * (kModelTuningProfile.UpperBound.LearningRate - kModelTuningProfile.LowerBound.LearningRate) * current_learning_rate + (float)current_gradient[2] * gradient_rate;
                trainerParam.BiasRate += (float)MLRandom.Shared.Range(-1.0, 1.0) * (kModelTuningProfile.UpperBound.BiasRate - kModelTuningProfile.LowerBound.BiasRate) * current_learning_rate + (float)current_gradient[3] * gradient_rate;
                trainerParam.Momentum += (float)MLRandom.Shared.Range(-1.0, 1.0)  * (kModelTuningProfile.UpperBound.Momentum - kModelTuningProfile.LowerBound.Momentum) * current_learning_rate + (float)current_gradient[4] * gradient_rate;
                trainerParam.WeightDecay += (float)MLRandom.Shared.Range(-1.0, 1.0) * (kModelTuningProfile.UpperBound.WeightDecay - kModelTuningProfile.LowerBound.WeightDecay) * current_learning_rate + (float)current_gradient[5] * gradient_rate;

                trainerParam.Epochs = Math.Clamp(trainerParam.Epochs, kModelTuningProfile.LowerBound.Epochs, kModelTuningProfile.UpperBound.Epochs);
                trainerParam.BatchSize = Math.Clamp(trainerParam.BatchSize, kModelTuningProfile.LowerBound.BatchSize, kModelTuningProfile.UpperBound.BatchSize);
                trainerParam.LearningRate = Math.Clamp(trainerParam.LearningRate, kModelTuningProfile.LowerBound.LearningRate, kModelTuningProfile.UpperBound.LearningRate);
                trainerParam.BiasRate = Math.Clamp(trainerParam.BiasRate, kModelTuningProfile.LowerBound.BiasRate, kModelTuningProfile.UpperBound.BiasRate);
                trainerParam.Momentum = Math.Clamp(trainerParam.Momentum, kModelTuningProfile.LowerBound.Momentum, kModelTuningProfile.UpperBound.Momentum);
                trainerParam.WeightDecay = Math.Clamp(trainerParam.WeightDecay, kModelTuningProfile.LowerBound.WeightDecay, kModelTuningProfile.UpperBound.WeightDecay);

                trainer_index++;
            }
        }

        NVector _previous_best_hpvec = new NVector(6);

        private void UpdateGenetic(KModelTuningProfile kModelTuningProfile, TTrainer[] trainers, float learning_rate, HyperparameterData[][] hyperparametersHistory, List<HyperparameterData> besthyperparameterDatas, int it_index, int iteration_best_score_index, int iteration_lowest_score_index)
        {
            // mutation rate is a chance of having 
            float mutation_rate = .25f;
            learning_rate = 1f;

            int trainer_index = 0;

            // init random values in the search space given by the profile
            foreach (var trainer in trainers)
            {
                var trainerParam = (trainer as IStochasticGradientDescentParameters);
                var sorted_hp = besthyperparameterDatas.OrderByDescending(t => t.Score).ToArray();
                var current_best_hpvec = (sorted_hp[0] as IStochasticGradientDescentParameters).GetHyperparameterVector();

                // first 2 runs as initial values
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

                    // decreasing lr if no better hp vec found on iteration (we probly overshoot the new best)
                    learning_rate *= .9f;
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

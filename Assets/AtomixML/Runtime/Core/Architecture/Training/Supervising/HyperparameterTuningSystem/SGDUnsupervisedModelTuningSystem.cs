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
    {
        private object _lock = new object();

        public struct HyperparameterData : IStochasticGradientDescentParameters
        {            
            public double Score { get; set; }

            public int Epochs { get; set; }
            public int BatchSize { get; set; }
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
            await Task.Delay(1);

            int it_index = 0;
            double[] scores = new double[trainers.Length];
            var bestHyperparameterDatas = new List<HyperparameterData>();

           /* var inputs = new TModelInput[t_inputs.Length];
            for(int i = 0; i < t_inputs.Length; ++i)
            {
                inputs[i] = JsonConvert.DeserializeObject<TModelInput>(JsonConvert.SerializeObject(t_inputs[i]));
            }*/

            while (it_index < iterations)
            {
                Debug.Log($"Tuner enter iteration {it_index}");

                // init random values in the search space given by the profile
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

                /* Parallel.For(0, trainers.Length, (index) =>
                 {
                     *//* var fit = trainers[index].Fit(t_inputs);
                      fit.RunSynchronously();
                      var tr_result = fit.Result;

                      var score = trainers[index].Score();
                      score.RunSynchronously();
                      var tr_score = score.Result;
                     *//*

                     var tr_result = trainers[index].FitSynchronously(t_inputs);
                     var tr_score = trainers[index].ScoreSynchronously();

                     lock (_lock)
                         scores[index] = tr_score;
                 });*/

                var fitTasks = trainers.Select(async (trainer, index) =>
                {
                    await Task.Delay(1);

                    // Fit the trainer asynchronously
                    var tr_result = trainer.FitSynchronously(t_inputs);

                    // Score the trainer asynchronously
                    var tr_score = trainer.ScoreSynchronously();

                    // Safely update the scores array
                    lock (_lock)
                    {
                        scores[index] = tr_score;
                    }
                }).ToArray();

                // Wait for all tasks to complete
                await Task.WhenAll(fitTasks);

                Debug.Log($"Tuner end fit iteration {it_index}");


                var best_score = double.MinValue;
                int best_score_index = -1;

                for (int i = 0; i < scores.Length; ++i)
                {
                    if (scores[i] > best_score)
                    {
                        best_score = scores[i];
                        best_score_index = i;
                    }
                }

                bestHyperparameterDatas.Add(new HyperparameterData(best_score, trainers[best_score_index] as IStochasticGradientDescentParameters));

                it_index++;
                await Task.Delay(1);
            }

            var best_overall_score = double.MinValue;
            int best_overall_score_index = -1;

            for (int i = 0; i < bestHyperparameterDatas.Count; ++i)
            {
                if (bestHyperparameterDatas[i].Score > best_overall_score)
                {
                    best_overall_score = scores[i];
                    best_overall_score_index = i;
                }
            }

            return bestHyperparameterDatas[best_overall_score_index];
        }
    }

}

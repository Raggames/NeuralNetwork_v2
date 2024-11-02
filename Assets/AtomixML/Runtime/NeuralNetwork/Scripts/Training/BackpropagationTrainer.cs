using Sirenix.OdinInspector;
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
    public class BackpropagationTrainer : NeuralNetworkTrainer
    {
        [Header("----PARAMS----")]
        public bool AutoStart = true;

        [Header("----NETWORK----")]
        public ModelBuilder Builder;
        public NeuralNetwork NeuralNetwork;

        // *************************************************************************************
        [Header("----MODE----")]
        public RunningMode Mode;
        public enum RunningMode
        {
            Train,
            Execute,
        }

        [Header("---- SAVING ----")]
        public bool Save;
        public int AutoSaveEveryEpochs = 5000;

       
        private WaitForSeconds delay;
        private Coroutine ExecutionCoroutine;

        [Header("----RUNTIME----")]

        /// <summary>
        /// Feedback on computation 'speed'
        /// </summary>
        [ReadOnly] public double epochs_per_second = 0;
        /// <summary>
        /// Avoid unity from freezing on compute large epoch values
        /// </summary>
        [ReadOnly] public float max_frame_time = 2;

        public double[][] x_datas;
        public double[][] t_datas;

        public double[] run_inputs;
        public double[] run_outputs;
        public double[] run_labels;
        private CancellationTokenSource _source;

        public void Start()
        {
            Initialize();

            if (!AutoStart)
                return;

            if (Mode == RunningMode.Train)
            {
                TrainAsync();
            }
            else
            {
                PredictAsync();
            }
        }

        private void PredictAsync()
        {
            Initialize();
            PrepareExecution();

            ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
        }

        [Button]
        private async void TrainAsync()
        {
            Initialize();
            PrepareTraining();

            _source = new CancellationTokenSource();
            await DoBackPropagationTraining(_source.Token);
        }

        [Button]
        private void Cancel()
        {
            _source.Cancel();
        }

        [Button]
        private void SaveModel()
        {
            SaveCurrentModel(NeuralNetwork);
        }

        private void OnGUI()
        {
           /* if (GUI.Button(new Rect(10, 10, 100, 30), "Train"))
            {
                PrepareTraining();

                ExecutionCoroutine = StartCoroutine(DoBackPropagationTraining());
            }
*/
            if (GUI.Button(new Rect(10, 100, 100, 30), "Load"))
            {
                PrepareExecution();
            }

            if (GUI.Button(new Rect(10, 150, 100, 30), "Test"))
            {
                ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
            }

            if (GUI.Button(new Rect(10, 50, 100, 30), "Save"))
            {
                SaveCurrentModel(NeuralNetwork);
            }
        }

        public virtual void Initialize()
        {
            NeuralNetwork = new NeuralNetwork();
            NeuralNetwork.CreateNetwork(Builder);
            TrainingSetting.Init();
        }

        public void PrepareTraining()
        {
            NeuralNetwork.SeedRandomWeights(InitialWeightRange.x, InitialWeightRange.y);
            InitializeTrainingBestWeightSet(NeuralNetwork);
        }

        public void PrepareExecution()
        {
            NeuralNetwork.InitializeModelFromSave(LoadDataByName(SaveName));
        }

        private IEnumerator DoExecuting(int runs)
        {
            correctRuns = 0;
            wrongRuns = 0;

            int count = 0;

            TrainingSetting.GetTrainDatas(out x_datas, out t_datas);

            int[] sequence_indexes = new int[x_datas.Length];
            for (int i = 0; i < sequence_indexes.Length; ++i)
                sequence_indexes[i] = i;

            Shuffle(sequence_indexes);

            for (int i = 0; i < runs; ++i)
            {
                if(i >= x_datas.Length)
                {
                    Debug.Log("All data have been treated");
                    break;
                }

                run_inputs = x_datas[sequence_indexes[i]];
                run_labels = t_datas[sequence_indexes[i]]; 
                
                NeuralNetwork.FeedForward(run_inputs, out run_outputs);
                ComputeAccuracy(run_labels, run_outputs);

                CurrentEpoch++;
                count++;
                yield return null;
            }

        }

        #region BackpropagationTraining

        private async Task DoBackPropagationTraining(CancellationToken cancellationToken)
        {
            CurrentEpoch = 0;

            double current_time = 0;

            // Get training datas from the setting
            TrainingSetting.GetTrainDatas(out x_datas, out t_datas);

            // Compute number of iterations 
            // Batchsize shouldn't be 0
            int iterations_per_epoch = x_datas.Length / BatchSize;

            int[] sequence_indexes = new int[x_datas.Length];
            for (int i = 0; i < sequence_indexes.Length; ++i)
                sequence_indexes[i] = i;

            for (int i = 0; i < Epochs; ++i)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int dataIndex = 0;

                // Shuffle datas each epoch
                Shuffle(sequence_indexes);

                // Going through all data batched, mini-batch or stochastic, depending on BatchSize value
                double error_sum = 0;
                for(int d = 0; d < iterations_per_epoch; ++d)
                {
                    for (int j = 0; j < BatchSize; ++j)
                    {
                        run_inputs = x_datas[sequence_indexes[dataIndex]];
                        run_labels = t_datas[sequence_indexes[dataIndex]];

                        NeuralNetwork.FeedForward(run_inputs, out run_outputs);

                        // we accumulate gradients each pass of the batch
                        NeuralNetwork.ComputeDenseGradients(run_labels, run_outputs);

                        // updating accuracy by simply count correct/uncorrect runs
                        ComputeAccuracy(run_labels, run_outputs);

                        error_sum += ComputeLossFunction(run_outputs, run_labels);
                        dataIndex++;

                        await Task.Delay(1);

                        /*current_time += Time.deltaTime;
                        if (current_time > max_frame_time)
                        {
                            current_time = 0;
                            //yield return null;
                        }*/
                    }

                    // Computing gradients average over batchsize
                    NeuralNetwork.MeanDenseGradients(BatchSize);
                    // Computing new weights and reseting gradients to 0 for next batch
                    NeuralNetwork.UpdateDenseWeights(LearningRate, Momentum, WeightDecay, BiasRate);
                }

                double last_mean_error = currentMeanError;

                // Computing the mean error
                currentMeanError = error_sum / x_datas.Length;

                if (currentMeanError < last_mean_error)
                {
                    // Keeping a trace of the best set we have
                    // if the accuracy falls appart at some point we have the best iteration avalaible for saving
                    MemorizeBestSet(NeuralNetwork, currentMeanError);
                }

                // If under target error, stop
                if (currentMeanError < Target_Mean_Error)
                {
                    break;
                }

                DecayLearningRate();


                CurrentEpoch++;
                //epochs_per_second = CurrentEpoch / Time.realtimeSinceStartup;
            }

            //NeuralNetwork.GetAndSaveWeights();
        }
             

        public void ExecuteFeedForward()
        {
            TrainingSetting.GetNextValues(out run_inputs, out run_labels);
            NeuralNetwork.FeedForward(run_inputs, out run_outputs);
        }

        public void DecayLearningRate()
        {
            LearningRate -= LearningRate * LearningRateDecay;
        }

        #endregion

    }
}

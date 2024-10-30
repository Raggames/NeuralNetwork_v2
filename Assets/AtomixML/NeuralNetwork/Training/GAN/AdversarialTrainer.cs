using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    public class AdversarialTrainer : NeuralNetworkTrainer
    {
        public BackpropagationTrainer DiscriminantNetworkTrainer;
        public BackpropagationTrainer GeneratorNetworkTrainer;

        public int DiscriminantPassesPerEpoch = 200;
        public int GeneratorPassesPerEpoch = 200;

        public int GeneratorCorrectRuns = 0;
        public int GeneratorWrongRuns = 0;
        public float GeneratorAccuracy = 0;

        public int DiscriminantCorrectRuns = 0;
        public int DiscriminantWrongRuns = 0;
        public float DiscriminantAccuracy = 0;

        public void Awake()
        {
            /*Discriminant = new NeuralNetwork();
            Discriminant.CreateNetwork(this, DiscriminantNetworkTrainer.Builder);
            Discriminant.LoadAndSetWeights(LoadDataByName(DiscriminantNetworkTrainer.SaveName));
            (TrainingSetting as WordGenerationTrainingSetting).DiscriminantNetwork = Discriminant;*/
        }

        private void Start()
        {
            DiscriminantNetworkTrainer.Initialize();
            DiscriminantNetworkTrainer.PrepareTraining();

            GeneratorNetworkTrainer.Initialize();
            GeneratorNetworkTrainer.PrepareTraining();

            StartCoroutine(TrainAdversarial());
        }

        public IEnumerator TrainAdversarial()
        {
            double[] discriminant_desired_output = new double[1] { 1 };

            for(int i = 0; i < Epochs; ++i)
            {
                double[] discriminant_run_output = new double[0];

                GeneratorNetworkTrainer.ExecuteFeedForward();

                DiscriminantNetworkTrainer.NeuralNetwork.FeedForward(GeneratorNetworkTrainer.run_outputs, out discriminant_run_output);

              /*  if (discriminant_run_output[0] > .92f)
                {
                    GeneratorCorrectRuns++;
                    DiscriminantWrongRuns++;
                    DiscriminantNetworkTrainer.GetLoss(discriminant_run_output, new double[] { 1 });

                    DiscriminantNetworkTrainer.NeuralNetwork.BackPropagate(DiscriminantNetworkTrainer.Current_Mean_Error, discriminant_run_output, new double[] { 0 }, DiscriminantNetworkTrainer.LearningRate, DiscriminantNetworkTrainer.Momentum, DiscriminantNetworkTrainer.WeightDecay, DiscriminantNetworkTrainer.BiasRate);
                }
                else
                {
                    GeneratorWrongRuns++;
                    DiscriminantCorrectRuns++;


                    GeneratorAccuracy = ((float)GeneratorCorrectRuns * 1) / (float)(GeneratorCorrectRuns + GeneratorWrongRuns); // ugly 2 - check for divide by zero
                    GeneratorAccuracy *= 100f;

                    DiscriminantAccuracy = ((float)DiscriminantCorrectRuns * 1) / (float)(DiscriminantCorrectRuns + DiscriminantWrongRuns); // ugly 2 - check for divide by zero
                    DiscriminantAccuracy *= 100f;

                    Debug.LogError("GEN_Outp => " + (DiscriminantNetworkTrainer.TrainingSetting as RealFalseWordRecognitionTrainingSetting).UnwrapWord(GeneratorNetworkTrainer.run_outputs));

                    // Because we are inputing a generator result in the discriminant, we want the discriminant to return something like 1

                    //double[] generator_t_values = GetGeneratorTestValues(discriminant_desired_output, discriminant_run_output);
                    var x_val = new double[0];
                    double[] generator_t_values = new double[0];

                    DiscriminantNetworkTrainer.TrainingSetting.GetNextValues(out generator_t_values, out x_val);
                    Debug.LogError("TVAL =>" + (DiscriminantNetworkTrainer.TrainingSetting as RealFalseWordRecognitionTrainingSetting).UnwrapWord(generator_t_values));
                 
                    double[] grad_inp = new double[0];

                    GeneratorNetworkTrainer.NeuralNetwork.ComputeGradients(generator_t_values, grad_inp);
                    // Update generator weights to allow him to trick the discriminant
                    GeneratorNetworkTrainer.NeuralNetwork.ComputeWeights(GeneratorNetworkTrainer.LearningRate, GeneratorNetworkTrainer.Momentum, GeneratorNetworkTrainer.WeightDecay, GeneratorNetworkTrainer.BiasRate);

                }*/

                yield return null;
            }
        }

        private double[] GetGeneratorTestValues(double[] discriminant_desired_output, double[] discriminant_run_output)
        {
            double[] discr_inp_grads = DiscriminantNetworkTrainer.NeuralNetwork.ComputeDenseGradients(discriminant_desired_output, discriminant_run_output);

            double[] generator_t_values = new double[DiscriminantNetworkTrainer.Builder.InputLayer.NeuronsCount];

            // Comment calculer ce que le generateur aurait du donner ?
            // On fait une remontée de gradients et puis ensuite ??
            //
            for (int k = 0; k < DiscriminantNetworkTrainer.Builder.InputLayer.NeuronsCount; ++k)
            {
                for (int g = 0; g < discr_inp_grads.Length; ++g)
                {
                    generator_t_values[k] += discr_inp_grads[g] * DiscriminantNetworkTrainer.NeuralNetwork.DenseLayers[0].Weights[k, g];
                }
                //discr_input_from_grads[k] /= discr_inp_grads.Length;
            }

            return generator_t_values;
        }

        public void RunReverse()
        {

        }

        /*public override float ComputeHeuristic(double[] result, double[] testValue)
        {
            double[] t_val = new double[1];
            Discriminant.FeedForward(result, out t_val);

            // if t_val = 1, the word generated is considered as a real word
            // 
            return 1f - (float)t_val[0];
        }*/

    }
}


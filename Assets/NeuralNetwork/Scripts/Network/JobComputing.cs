/*using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.NeuralNetwork.Scripts.Network
{
    class JobComputing
    {
        public void JobComputeFeedForward(double[] inputs, out double[] results)
        {
            double[] layerResult = new double[HiddenLayers[0].NeuronsCount];
            JobComputeLayer(inputs, out layerResult, _ihWeights, _hBiases, HiddenLayers[0]);
            JobComputeLayer(layerResult, out _outputs, hoWeights, oBiases, OutputLayer);

            results = _outputs;
        }

        private void JobComputeLayer(double[] inputs, out double[] outputs, double[,] weights, double[] bias, NeuralNetworkLayer layerData)
        {
            NativeArray<double> inputsJob = new NativeArray<double>(inputs.Length * layerData.NeuronsCount, Allocator.TempJob);
            NativeArray<double> layerOutputJob = new NativeArray<double>(weights.GetLength(0) * weights.GetLength(1), Allocator.TempJob);
            NativeArray<double> weightsJob = new NativeArray<double>(weights.GetLength(0) * weights.GetLength(1), Allocator.TempJob);

            int k = 0;
            for (int i = 0; i < weights.GetLength(0); ++i)
            {
                for (int j = 0; j < weights.GetLength(1); ++j)
                {
                    weightsJob[k++] = weights[i, j];
                }
            }

            for (int i = 0; i < inputsJob.Length; ++i)
            {
                inputsJob[i] = inputs[i % inputs.Length];
            }

            ComputeWeightsJob computeWeightsJob = new ComputeWeightsJob
            {
                Inputs = inputsJob,
                Outputs = layerOutputJob,
                Weights = weightsJob,
            };

            JobHandle handle = computeWeightsJob.Schedule(weightsJob.Length, 1);

            handle.Complete();

            double[] results = new double[layerData.NeuronsCount];

            int index = 0;
            int pointer = 0;
            for (int i = 0; i < layerOutputJob.Length; ++i)
            {
                if (index < inputs.Length)
                {
                    results[pointer] += layerOutputJob[i];
                    index++;
                }
                if (index == inputs.Length)
                {
                    double value = results[pointer];
                    value += bias[pointer];
                    value /= inputs.Length;

                    if (layerData.ActivationFunction != ActivationFunctions.Softmax)
                    {
                        value = JNNMath.ComputeActivation(layerData.ActivationFunction, false, value);
                    }
                    results[pointer] = value;

                    index = 0;
                    pointer++;
                }
            }
            if (layerData.ActivationFunction == ActivationFunctions.Softmax)
            {
                results = Softmax(results);
            }
            outputs = results;

            layerOutputJob.Dispose();
            inputsJob.Dispose();
            weightsJob.Dispose();
        }
#endregion
    }
}
*/
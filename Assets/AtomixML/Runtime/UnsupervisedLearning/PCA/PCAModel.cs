using Atom.MachineLearning.Core;
using System;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.PCA
{
    [Serializable]
    public class PCAModel :
        IMLModel<NVector, NVector>,
        IMLPipelineElement<NVector, NVector>
    {
        public string ModelName { get; set; } = "PCASimple";
        public string ModelVersion { get; set; } = "1.0.0";

        [MachineLearnedParameter, SerializeField] public NMatrix _projectionMatrix;
        [MachineLearnedParameter, SerializeField] public NVector _meanVector;
        [MachineLearnedParameter, SerializeField] public NVector _stdDeviationVector;

        // for transformation of input data
        protected NMatrix projectionMatrix { get => _projectionMatrix; private set => _projectionMatrix = value; }

        // for standardisation of input data
        protected NVector meanVector { get => _meanVector; private set => _meanVector = value; }
        protected NVector stdDeviationVector { get => _stdDeviationVector; private set => _stdDeviationVector = value; }

        /// <summary>
        /// Initialize the learned parameters
        /// Usually called by the trainer after training
        /// </summary>
        /// <param name="projectionMatrix"></param>
        /// <param name="meanVector"></param>
        /// <param name="stdDeviationVector"></param>
        public void Initialize(NMatrix projectionMatrix, NVector meanVector, NVector stdDeviationVector)
        {
            this.projectionMatrix = projectionMatrix;
            this.meanVector = meanVector;
            this.stdDeviationVector = stdDeviationVector;
        }

        /// <summary>
        /// Using the projection matrix to compress the input data to a lower dimension data
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public NVector Predict(NVector inputData)
        {
            // input data should be standardized with the same mean and deviation from the learning process
            var standardizedInput = NVector.Standardize(inputData, meanVector, stdDeviationVector, stdDeviationVector.Average());

            return standardizedInput * projectionMatrix;    
        }

        /// <summary>
        /// Using the transposed projection matrix to 'decompress' the predicted data to higher dimensions
        /// </summary>
        /// <param name="predictedData"></param>
        /// <returns></returns>
        public NVector Decompress(NVector predictedData)
        {
            if (predictedData.Length != projectionMatrix.Columns) throw new System.Exception($"The input vector should be an output of the prediction. " +
                $"It should have a number of features equals to the number of components (aka the reduced dimensions from input data");

            var transposed = NMatrix.Transpose(projectionMatrix);
            return predictedData * transposed;
        }
    }

}

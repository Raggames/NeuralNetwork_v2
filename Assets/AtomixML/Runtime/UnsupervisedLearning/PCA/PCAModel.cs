
using Atom.MachineLearning.Core;

namespace Atom.MachineLearning.Unsupervised.PCA
{
    public class PCAModel :
        IMLModel<NVector, NVector>,
        IMLPipelineElement<NVector, NVector>
    {
        public string AlgorithmName => "PCASimple";

        // for transformation of input data
        protected NMatrix projectionMatrix { get; private set; }

        // for standardisation of input data
        protected NVector meanVector { get; private set; }
        protected NVector stdDeviationVector { get; private set; }  

        public void Initialize(NMatrix projectionMatrix, NVector meanVector, NVector stdDeviationVector)
        {
            this.projectionMatrix = projectionMatrix;
            this.meanVector = meanVector;
            this.stdDeviationVector = stdDeviationVector;
        }

        public NVector Predict(NVector inputData)
        {
            // input data should be standardized with the same mean and deviation from the learning process
            var standardizedInput = NVector.Standardize(inputData, meanVector, stdDeviationVector);

            return projectionMatrix * standardizedInput;    
        }


        public void Load(string filename)
        {
            throw new System.NotImplementedException();
        }

        public void Save(string outputFilename)
        {
            throw new System.NotImplementedException();
        }
    }

}

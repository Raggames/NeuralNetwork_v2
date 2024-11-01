
using Atom.MachineLearning.Core;

namespace Atom.MachineLearning.Unsupervised.PCA
{
    public class PCAModel :
        IMLModel<NVector, NVector>,
        IMLPipelineElement<NVector, NVector>
    {
        public string AlgorithmName => "PCASimple";

        protected NMatrix projectionMatrix { get; private set; }

        // for standardisation of input data
        protected double[] means { get; private set; }
        protected double[] stdDeviations { get; private set; }  

        //public void Initialize(NMatrix projectionMatrix, )

        public NVector Predict(NVector inputData)
        {
            // input data should be standardized with the same mean and deviation from the learning process
            var standardizedInput = NVector.Standardize(inputData, means, stdDeviations);

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


using Atom.MachineLearning.Core;

namespace Atom.MachineLearning.Unsupervised.PCA
{
    public class PCAModel :
        IMLModel<NVector, NVector>,
        IMLPipelineElement<NVector, NVector>
    {
        public string AlgorithmName => "PCASimple";

        private double[] _means;
        private double[] _stdDeviations;

        public NVector Predict(NVector inputData)
        {
            // the input data has been standardised by the runner


            throw new System.NotImplementedException();
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

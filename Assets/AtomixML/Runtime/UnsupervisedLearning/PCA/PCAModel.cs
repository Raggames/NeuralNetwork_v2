
using Atom.MachineLearning.Core;

namespace Atom.MachineLearning.Unsupervised.PCA
{
    public class PCAModel :
        IMLModel<NVector, NVector>,
        IMLPipelineElement<NVector, NVector>
    {
        public string AlgorithmName => "PCASimple";

        public NMatrix ProjectionMatrix { get; private set; }

        public NVector Predict(NVector inputData)
        {
            // input data should be standardized
            return ProjectionMatrix * inputData;    
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

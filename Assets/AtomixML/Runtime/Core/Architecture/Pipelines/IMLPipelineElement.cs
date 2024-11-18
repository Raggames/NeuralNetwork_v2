using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// A pipeline element can be either a model, a transformation layer, etc...
    /// </summary>
    public interface IMLPipelineElement<T, K> 
    {
        public K Predict(T inputData);
    }
}

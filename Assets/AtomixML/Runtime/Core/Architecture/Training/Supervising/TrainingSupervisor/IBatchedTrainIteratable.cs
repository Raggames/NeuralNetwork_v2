namespace Atom.MachineLearning.Core.Training
{
    public interface IBatchedTrainIteratable : IEpochIteratable
    {
        public void OnTrainNextBatch(int[] indexes);
    }
}

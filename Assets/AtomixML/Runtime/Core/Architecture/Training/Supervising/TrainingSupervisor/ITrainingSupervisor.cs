using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Training
{
    public interface ITrainingSupervisor
    {
        public ITrainingSupervisor SetEpochIteration(IEpochIteratable target);
        public ITrainingSupervisor SetTrainIteration(ITrainIteratable target);
        public ITrainingSupervisor SetTrainBatchIteration(IBatchedTrainIteratable target);
        public ITrainingSupervisor SetAutosave(int epoch_interval = 1);
        
        public void Cancel();
        public Task RunOnlineAsync(int epochs, int trainLenght = 0, bool shuffleTrainIndex = true);
        public Task RunBatchedAsync(int epochs, int trainLenght = 0, int batchSize = 5, bool shuffleTrainIndex = true);

    }
}

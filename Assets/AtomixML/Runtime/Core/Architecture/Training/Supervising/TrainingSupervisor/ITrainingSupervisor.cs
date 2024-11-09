using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Training
{
    public interface ITrainingSupervisor
    {
        public ITrainingSupervisor SetEpochIteration(IEpochIteratable target);
        public ITrainingSupervisor SetTrainIteration(ITrainIteratable target);

        public void Cancel();
        public Task RunAsync(int epochs, int trainLenght = 0, bool shuffleTrainIndex = true);

    }
}

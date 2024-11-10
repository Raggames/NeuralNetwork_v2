using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Training
{
    public class ParallelTrainingSupervisor : ITrainingSupervisor
    {
        public void Cancel()
        {
            throw new NotImplementedException();
        }

        public Task RunAsync(int epochs, int trainLenght = 0, bool shuffleTrainIndex = true)
        {
            throw new NotImplementedException();
        }

        public ITrainingSupervisor SetEpochIteration(IEpochIteratable target)
        {
            throw new NotImplementedException();
        }

        public ITrainingSupervisor SetTrainIteration(ITrainIteratable target)
        {
            throw new NotImplementedException();
        }
    }
}

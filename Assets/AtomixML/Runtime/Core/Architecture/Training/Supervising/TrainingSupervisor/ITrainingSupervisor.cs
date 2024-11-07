using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Training
{
    public interface ITrainingSupervisor<T>
    {
        public T SetEpochIteration(IEpochIteratable target);
        public T SetTrainIteration(ITrainIteratable target);
    }
}

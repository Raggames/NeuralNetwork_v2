using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Training
{
    public interface IEpochIteratable
    {
        public void OnBeforeEpoch(int epochIndex);
        public void OnAfterEpoch(int epochIndex);
    }
}

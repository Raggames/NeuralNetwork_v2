using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    public interface IGradientDescentOptimizer
    {
        protected EpochSupervisorAsync epochSupervisor { get; set; }
    }
}

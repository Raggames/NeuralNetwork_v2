using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.UtilityBasedAgents
{
    public interface IUtility
    {
        public double ComputeWeight(AIAgent agent);
    }
}

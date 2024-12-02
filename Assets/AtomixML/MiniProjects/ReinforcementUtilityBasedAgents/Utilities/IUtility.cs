using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.ReinforcementUtilityBasedAgents
{
    public interface IUtility
    {
        public double ComputeWeight(AIAgent agent);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.ReinforcementUtilityBasedAgents
{
    public interface IAction
    {
        /// <summary>
        /// Each action has a set of utilities that will determine the rank of the action
        /// The best ranked aciton will be used each turn
        /// </summary>
        public List<IUtility> Utilities { get; }

        public double Rank(AIAgent agent);
        public Task Execute(AIAgent agent);
    }
}

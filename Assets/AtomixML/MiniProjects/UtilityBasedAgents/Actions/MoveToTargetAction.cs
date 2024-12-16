using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.UtilityBasedAgents
{
    /// <summary>
    /// An action where the agent move to a potential target.
    /// The target computation is an utility subject to parameters from any in range target
    /// </summary>
    public class MoveToTargetAction : ActionBehaviourBase
    {
        public override List<IUtility> Utilities => throw new NotImplementedException();

        public override Task Execute(AIAgent agent)
        {
            throw new NotImplementedException();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// Just to identitfy a field that represent a parameter of the model (aka a that has been learnt)
    /// </summary>
    [AttributeUsage(AttributeTargets.Field)]
    public class MachineLearnedParameterAttribute : Attribute
    {
    }
}

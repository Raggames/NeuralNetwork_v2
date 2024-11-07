using System;

namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// Just to identitfy a field that represent a parameter of the model
    /// </summary>
    [AttributeUsage(AttributeTargets.Field)]
    public class HyperParameterAttribute : Attribute
    {
    }
}

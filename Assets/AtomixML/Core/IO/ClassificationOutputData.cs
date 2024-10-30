using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// An output data as an integer 'ClassLabel' made for classification outputs
    /// </summary>
    public class ClassificationOutputData : IMLOutputData
    {
        public int ClassLabel { get; set; }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Training
{
    /// <summary>
    /// A profile of hyperparameters 
    /// </summary>
    public interface ITuningProfile<THyperParameterSet> where THyperParameterSet : IHyperParameterSet
    {
        /// <summary>
        /// Lower value for each hyper parameter
        /// </summary>
        public THyperParameterSet LowerBound { get; }
        public THyperParameterSet UpperBound { get;}
    }
}

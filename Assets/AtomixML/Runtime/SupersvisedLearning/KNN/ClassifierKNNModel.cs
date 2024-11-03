using Assets.AtomixML.Runtime.SupersvisedLearning.KNN;
using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Supervised.KNN
{
    public class ClassifierKNNModel : IKNNModel
    {
        [HyperParameter] private bool _useNeighboorDistancePonderation;

        /// <summary>
        /// The neighboor matrix is a Mxn matrix where : 
        /// M is the features count + 1 label column 
        /// n is the total number of neighboors
        /// </summary>
        [LearnedParameter] private NMatrix _neighboorsMatrix;

        /// <summary>
        /// A Mxn matrix representing a 0-1 value for the distribution of every label        
        /// </summary>
        [LearnedParameter] private NMatrix _neighboorDistributionMatrix;
    }
}

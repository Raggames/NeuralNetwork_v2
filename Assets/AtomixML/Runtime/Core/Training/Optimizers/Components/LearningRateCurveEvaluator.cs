using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Trainig.Optimizers
{
    [Serializable]
    /// <summary>
    /// Allows to compute a learning rate by defining a curve over time (epochs).
    /// </summary>
    public class LearningRateCurveEvaluator
    {
        [SerializeField] private AnimationCurve _learningRateCurve;
    }
}

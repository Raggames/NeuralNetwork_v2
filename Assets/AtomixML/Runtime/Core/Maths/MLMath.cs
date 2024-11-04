using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Maths
{
    public static class MLMath
    {
        public static double Gaussian(double distance, double radius)
        {
            return Math.Exp(-distance / (2 * radius * radius));
        }

    }
}

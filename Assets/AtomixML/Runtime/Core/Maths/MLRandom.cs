using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Maths
{
    public static class MLRandom
    {
        private static Random _shared;

        public static Random Shared
        {
            get
            {
                if(_shared == null)
                {
                    _shared = new System.Random(Guid.NewGuid().GetHashCode());
                }

                return _shared;
            }
        }

        public static double Range(this Random random, double minValue, double maxValue)
        {
            return random.NextDouble() * (maxValue - minValue) + minValue;
        }
        
        public static int Range(this Random random, int minValue, int maxValue)
        {
            if (minValue > maxValue)
                return minValue;

            return random.Next(minValue, maxValue);
        }
    }
}

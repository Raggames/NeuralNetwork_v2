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
    }
}

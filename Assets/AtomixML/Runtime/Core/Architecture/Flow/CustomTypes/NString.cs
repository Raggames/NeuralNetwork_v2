using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    [Serializable]
    public struct NString : IMLInOutData
    {
        [ShowInInspector, ReadOnly] public string[] Data { get; set; }

        public int Length => Data.Length;

        public string this[int index]
        {
            get
            {
                return Data[index];
            }
            set
            {
                Data[index] = value;
            }
        }
    }
}

using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    [Serializable]
    public struct NStringVector : IMLInOutData
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

        public NStringVector(string[] data)
        {
            Data = data;
        }

        public NStringVector(int length)
        {
            Data = new string[length];
        }

    }

    public static class NStringVectorExtensions
    {
        public static NStringVector[] ToNStringVectorArray(this string[,] datas)
        {
            var nstring = new NStringVector[datas.GetLength(0)];
            int width = datas.GetLength(1);
            for (int i = 0; i < datas.GetLength(0); ++i)
            {
                var data = new string[width];
                for (int j = 0; j < width; ++j)
                    data[j] = datas[i, j];

                nstring[i] = new NStringVector(data);
            }

            return nstring;
        }

    }
}

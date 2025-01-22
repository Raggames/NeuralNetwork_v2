﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Training
{
    public interface ITrainIteratable : IEpochIteratable
    {
        public void OnTrainNext(int index);
    } 
}

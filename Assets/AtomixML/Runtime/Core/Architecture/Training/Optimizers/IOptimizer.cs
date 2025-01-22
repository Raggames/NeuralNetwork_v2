using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimization
{
    public interface IOptimizer<T, TInput, TOuput> where T : IOptimizable<TInput, TOuput>
    {
        public Task<T> OptimizeAsync(T model);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.SerializationHelper
{
    public static class ArrayHelpers
    {
        public static float[][] ConvertToJaggedArray(float[,] multiArray)
        {
            int rows = multiArray.GetLength(0);
            int cols = multiArray.GetLength(1);
            float[][] jaggedArray = new float[rows][];
            for (int i = 0; i < rows; i++)
            {
                jaggedArray[i] = new float[cols];
                for (int j = 0; j < cols; j++)
                {
                    jaggedArray[i][j] = multiArray[i, j];
                }
            }
            return jaggedArray;
        }
    }
}

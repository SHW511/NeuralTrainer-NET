using System.Collections.Generic;
using System.Text.Json.Serialization;
using System;
using System.Runtime.Serialization;
using System.Xml.Serialization;
using System.Xml;
using System.Xml.Schema;

namespace NeuralNetwork
{
    //[Serializable]
    public abstract class Layer // : IXmlSerializable
    {
        public bool Built { get; set; } = false;
        public int InputDim { get; set; }
        public int OutputDim { get; set; }
        public int[] InputShape { get; set; }
        public float[,] Weights { get; set; }

        public abstract void Build(int[] inputShape);
        public abstract float[,] Call(float[,] inputs);
        public abstract float[,,,] Call(float[,,,] inputs);
        public abstract int[] GetOutputShape(int[] inputShape);
        public virtual Dictionary<string, object> GetConfig()
        {
            return new Dictionary<string, object>();
        }
        public abstract float[,] Backward(float[,] gradient);
        public abstract float[,,,] Backward(float[,,,] gradient);
        public XmlSchema GetSchema() => null;

        public Layer() { }

        protected Layer(SerializationInfo info, StreamingContext context)
        {
            Built = info.GetBoolean("Built");
            InputDim = info.GetInt32("InputDim");
            OutputDim = info.GetInt32("OutputDim");
            InputShape = (int[])info.GetValue("InputShape", typeof(int[]));
            Weights = ConvertToMultidimensionalArray((float[][])info.GetValue("Weights", typeof(float[][])));
        }

        private float[][] ConvertToJaggedArray(float[,] multiArray)
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

        private float[,] ConvertToMultidimensionalArray(float[][] jaggedArray)
        {
            int rows = jaggedArray.Length;
            int cols = jaggedArray[0].Length;
            float[,] multiArray = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    multiArray[i, j] = jaggedArray[i][j];
                }
            }
            return multiArray;
        }

        public void WriteXml(XmlWriter writer)
        {
            writer.WriteElementString("Built", Built.ToString());
            writer.WriteElementString("InputDim", InputDim.ToString());
            writer.WriteElementString("OutputDim", OutputDim.ToString());
            writer.WriteStartElement("InputShape");
            foreach (var dim in InputShape)
            {
                writer.WriteElementString("Dim", dim.ToString());
            }
            writer.WriteEndElement();
            writer.WriteStartElement("Weights");
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                writer.WriteStartElement("Row");
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    writer.WriteElementString("Col", Weights[i, j].ToString());
                }
                writer.WriteEndElement();
            }
            writer.WriteEndElement();
        }

        public void ReadXml(XmlReader reader)
        {
            reader.ReadStartElement("Layer");
            Built = bool.Parse(reader.ReadElementString("Built"));
            InputDim = int.Parse(reader.ReadElementString("InputDim"));
            OutputDim = int.Parse(reader.ReadElementString("OutputDim"));
            reader.ReadStartElement("InputShape");
            var inputShapeList = new List<int>();
            while (reader.NodeType != XmlNodeType.EndElement)
            {
                inputShapeList.Add(int.Parse(reader.ReadElementString("Dim")));
            }
            InputShape = inputShapeList.ToArray();
            reader.ReadEndElement();
            reader.ReadStartElement("Weights");
            var weightsList = new List<float[]>();
            while (reader.NodeType != XmlNodeType.EndElement)
            {
                reader.ReadStartElement("Row");
                var rowList = new List<float>();
                while (reader.NodeType != XmlNodeType.EndElement)
                {
                    rowList.Add(float.Parse(reader.ReadElementString("Col")));
                }
                weightsList.Add(rowList.ToArray());
                reader.ReadEndElement();
            }
            Weights = ConvertToMultidimensionalArray(weightsList.ToArray());
            reader.ReadEndElement();
        }
    }
}

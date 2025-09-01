package com.mycompany.concu_proyecto;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordFreq {

    // =======================
    // Mapper UNIGRAM (palabras)
    // =======================
    public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private final Text word = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().toLowerCase()
                    .replaceAll("[^\\p{L}\\p{Nd}\\s]+", " "); // deja letras/números/espacios
            StringTokenizer itr = new StringTokenizer(line);

            while (itr.hasMoreTokens()) {
                String w = itr.nextToken();
                if (!w.isEmpty()) {
                    word.set(w);
                    context.write(word, ONE);
                }
            }
        }
    }

    // =======================
    // Mapper BIGRAM (pares consecutivos)
    // =======================
    public static class BigramMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private final Text bigram = new Text();

        // Para unir última palabra de la línea anterior con la primera de la actual (mismo split)
        private String prevLast = null;

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().toLowerCase()
                    .replaceAll("[^\\p{L}\\p{Nd}\\s]+", " ");
            String[] words = line.trim().isEmpty() ? new String[0] : line.trim().split("\\s+");

            // Bigrama que cruza líneas (mismo split)
            if (prevLast != null && words.length > 0) {
                bigram.set(prevLast + " " + words[0]);
                context.write(bigram, ONE);
            }

            // Bigramas dentro de la misma línea
            for (int i = 0; i < words.length - 1; i++) {
                if (!words[i].isEmpty() && !words[i + 1].isEmpty()) {
                    bigram.set(words[i] + " " + words[i + 1]);
                    context.write(bigram, ONE);
                }
            }

            // guarda la última palabra para encadenar con la siguiente línea
            if (words.length > 0) prevLast = words[words.length - 1];
        }
    }

    // =======================
    // Reducer (y Combiner)
    // =======================
    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable result = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) sum += v.get();
            result.set(sum);
            context.write(key, result);
        }
    }

    // =======================
    // MAIN (acepta 2 o 3 argumentos)
    // =======================
    public static void main(String[] args) throws Exception {

        // Permite:
        //  - 3 args: <unigram|bigram> <input> <output>
        //  - 2 args: <input> <output>  (modo compat -> unigram)
        if (args.length < 2) {
            System.err.println("Uso:");
            System.err.println("  WordFreq <unigram|bigram> <input> <output>");
            System.err.println("  WordFreq <input> <output>   (asume 'unigram')");
            System.exit(2);
        }

        final String mode;
        final Path input;
        final Path output;

        if (args.length == 2) {
            mode = "unigram";
            input = new Path(args[0]);
            output = new Path(args[1]);
        } else {
            mode = args[0].toLowerCase();
            input = new Path(args[1]);
            output = new Path(args[2]);
        }

        if (input.toString().equals(output.toString())) {
            throw new IllegalArgumentException("Input y output no pueden ser la misma ruta.");
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word-frequency-" + mode);
        job.setJarByClass(WordFreq.class);

        // Mapper / Reducer según modo
        switch (mode) {
            case "unigram":
                job.setMapperClass(TokenizerMapper.class);
                break;
            case "bigram":
                job.setMapperClass(BigramMapper.class);
                break;
            default:
                System.err.println("Modo desconocido: " + mode + " (use 'unigram' o 'bigram')");
                System.exit(2);
        }
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, input);     // INPUT = args[1] (o args[0] en modo compat)
        FileOutputFormat.setOutputPath(job, output);  // OUTPUT = args[2] (o args[1] en modo compat)

        System.out.println("Mode=" + mode + " | input=" + input + " | output=" + output);

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
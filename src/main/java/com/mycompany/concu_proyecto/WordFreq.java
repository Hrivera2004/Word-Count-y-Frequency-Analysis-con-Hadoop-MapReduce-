package com.mycompany.concu_proyecto;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordFreq {

    public static class Unicas_map extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static final Pattern PATRON = Pattern.compile("\\p{L}[\\p{L}\\p{Nd}]+");
        private static final IntWritable CONTADOR = new IntWritable(1);
        private final Text unica = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            Matcher matcher = PATRON.matcher(line);
            while (matcher.find()) {
                unica.set(matcher.group());
                context.write(unica, CONTADOR);
            }
        }
    }

    public static class Pares_map extends Mapper<LongWritable, Text, Text, IntWritable> {

        private static final Pattern PATRON = Pattern.compile("\\p{L}[\\p{L}\\p{Nd}]+");
        private static final IntWritable CONTADOR = new IntWritable(1);
        private final Text pares = new Text();

        private String prev = null;

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            Matcher matcher = PATRON.matcher(line);

            while (matcher.find()) {
                String word = matcher.group();
                if (prev != null) {
                    pares.set(prev + " " + word);
                    context.write(pares, CONTADOR);
                }
                prev = word;
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private static final IntWritable CONTADOR = new IntWritable(1);
        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) {
                sum += v.get();
            }

            if (sum >= 5000) {
                CONTADOR.set(sum);
                context.write(key, CONTADOR);
            }
        }
    }
    
    public static class MiPartir extends Partitioner<Text, IntWritable> {
        //Maybe use frequency table to distribute buckets by using spark
        @Override
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            char firstChar = key.toString().toLowerCase().charAt(0);

            if (firstChar >= 'a' && firstChar <= 'f') {
                return 0 % numReduceTasks;
            } else if (firstChar >= 'g' && firstChar <= 'l') {
                return 1 % numReduceTasks;
            } else if (firstChar >= 'm' && firstChar <= 'r') {
                return 2 % numReduceTasks;
            } else if (firstChar >= 's' && firstChar <= 'z') {
                return 3 % numReduceTasks;
            } else {
                return 0 % numReduceTasks;
            }
        }
    }

    public static void main(String[] args) throws Exception {

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

        job.setNumReduceTasks(4);
        switch (mode) {
            case "unigram":
                job.setMapperClass(Unicas_map.class);
                break;
            case "bigram":
                job.setMapperClass(Pares_map.class);
                break;
            default:
                System.exit(2);
        }
        job.setPartitionerClass(MiPartir.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

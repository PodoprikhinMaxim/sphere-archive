import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;

public class ConvertingDataJob extends Configured implements Tool {
    public static class ConvertingDataMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String out = value.toString().split("\t")[0];
            String in = value.toString().split("\t")[1];
            context.write(new LongWritable(Long.valueOf(out)), new LongWritable(Long.valueOf(in)));
        }
    }

    public static class ConvertingDataReducer extends Reducer <LongWritable, LongWritable, LongWritable, Text> {
        @Override
        protected void reduce(LongWritable key, Iterable<LongWritable> nums, Context context) throws IOException, InterruptedException {

            String str = "";
            str += Double.toString(1. / 4847571) + "\t"; //564549
            String pref = "";
            for(LongWritable i: nums) {
                str += pref + Long.toString(i.get());
                pref = ";";
            }
            context.write(key, new Text(str));
        }
    }

    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());

        job.setJarByClass(ConvertingDataJob.class);
        job.setJobName(ConvertingDataJob.class.getCanonicalName());

        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputValueClass(LongWritable.class);
        job.setMapOutputKeyClass(LongWritable.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.setMapperClass(ConvertingDataMapper.class);
        job.setReducerClass(ConvertingDataReducer.class);
        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new ConvertingDataJob(), args);
        System.exit(ret);
    }



}
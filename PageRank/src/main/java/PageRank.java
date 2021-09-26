import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
//import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;


public class PageRank {//extends Configured implements Tool {
    private static final String LINKS_SEPARATOR = "|";
    private static final double DAMPING = 0.85;
    private static NumberFormat NF = new DecimalFormat("00");
    private static int ITERATIONS = 7;
    //private static int flag = 0;

    public static class PageRankMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String link_in = value.toString().split("\t")[0];
            //if(value.toString().split("\t").length != 3){
            //    System.out.println(value.toString());
            //}
           // if(link_in == "1") {
           //     flag = 1;
            //}
            //if(flag == 1) {
            //    System.out.println(value.toString());
            //}
            double PR_value = Double.parseDouble(value.toString().split("\t")[1]);
            String links_out;
            if(value.toString().split("\t").length != 3){
                links_out = "";
            } else {
                links_out = value.toString().split("\t")[2];
            }
            //String links_out = value.toString().split("\t")[2];
            String[] links = links_out.split(";");

            for(String link: links) {
                if(link.equals("")) {
                    continue;
                }
                context.write(new LongWritable(Long.valueOf(link)), new Text(Double.toString(PR_value / links.length)));
            }
            context.write(new LongWritable(Long.valueOf(link_in)), new Text(PageRank.LINKS_SEPARATOR + links_out));
        }
    }

    public static class PageRankReducer extends Reducer<LongWritable, Text, LongWritable, Text> {
        @Override
        protected void reduce(LongWritable key, Iterable<Text> nums, Context context) throws IOException, InterruptedException {
            String links = "";
            double sumofPR = 0.0;
            for (Text value: nums) {
                String str = value.toString();
                if (str.startsWith(PageRank.LINKS_SEPARATOR)) {
                    if(str.split("\\|").length != 2) {
                        links = "";
                    } else {
                        links += str.split("\\|")[1];
                    }
                } else {
                    sumofPR += Double.parseDouble(value.toString());
                }

            }
            sumofPR = PageRank.DAMPING * sumofPR + (1 - PageRank.DAMPING) * (1 / 564549.);
            context.write(key, new Text(sumofPR + "\t" + links));
        }
    }

    private boolean job(String in, String out) throws IOException,
            ClassNotFoundException,
            InterruptedException {

        Job job = Job.getInstance(new Configuration());
        job.setJarByClass(PageRank.class);

        FileInputFormat.setInputPaths(job, new Path(in));
        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setMapperClass(PageRankMapper.class);

        FileOutputFormat.setOutputPath(job, new Path(out));
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);
        job.setReducerClass(PageRankReducer.class);

        return job.waitForCompletion(true);

    }
    public static void main(String[] args) throws Exception {

        PageRank pagerank = new PageRank();

        String OUT_PATH = "output_PR";
        
        for (int runs = 0; runs < ITERATIONS; runs++) {
            String inPath = OUT_PATH + "/iter" + NF.format(runs);
            String lastOutPath = OUT_PATH + "/iter" + NF.format(runs + 1);
            System.out.println("Running Job [" + (runs + 1) + "/" + PageRank.ITERATIONS + "] (PageRank calculation) ...");
            boolean isCompleted = pagerank.job(inPath, lastOutPath);
            if (!isCompleted) {
                System.exit(1);
            }
        }
        System.out.println("DONE!");
        System.exit(0);
    }
    
}

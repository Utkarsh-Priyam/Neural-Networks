package me.utkarshpriyam.Network;

import java.io.*;

public class Test
{

   public static void main(String[] args)
   {
      args = new String[] {"", ""};
      for (int i = 1; i <= 50; i++)
      {
         try
         {
            FileWriter fw = new FileWriter(new File("networkConfig.txt"));
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter pw = new PrintWriter(bw);

            pw.println("100-100-100");
            pw.println("train");
            pw.println("1");

            double d1 = Math.random() * 1.0, d2;
            pw.println(d1 + " 1.01 0.0 -1.0");
            pw.println("0.0");
            pw.println("100");

            d1 = -1 + Math.random() * 2;
            d2 = -1 + Math.random() * 2;
            pw.println(Math.min(d1, d2) + " " + Math.max(d1, d2));
            pw.println("false");

            pw.close();
            bw.close();
            fw.close();
         } catch (IOException ignored) {}

         Main.main(args);

         try
         {
            File file = new File("Output/Case " + i);
            file.mkdirs();

            file = new File("Output/Case " + i + "/pelValues.txt");
            file.createNewFile();

            FileWriter fw = new FileWriter(file);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter pw = new PrintWriter(bw);

            BufferedReader br = new BufferedReader(new FileReader("files/outputDump.txt"));
            String line = br.readLine();
            while (line != null)
            {
               pw.println(line);
               line = br.readLine();
            }

            pw.close();
            bw.close();
            fw.close();
         } catch (IOException ignored) {}

         try
         {
            File file = new File("Output/Case " + i + "/weights.txt");
            file.createNewFile();

            FileWriter fw = new FileWriter(file);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter pw = new PrintWriter(bw);

            BufferedReader br = new BufferedReader(new FileReader("files/weightDump.txt"));
            String line = br.readLine();
            while (line != null)
            {
               pw.println(line);
               line = br.readLine();
            }

            pw.close();
            bw.close();
            fw.close();
         } catch (IOException ignored) {}

         try
         {
            File file = new File("Output/Case " + i + "/config.txt");
            file.createNewFile();

            FileWriter fw = new FileWriter(file);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter pw = new PrintWriter(bw);

            BufferedReader br = new BufferedReader(new FileReader("networkConfig.txt"));
            String line = br.readLine();
            while (line != null)
            {
               pw.println(line);
               line = br.readLine();
            }

            pw.close();
            bw.close();
            fw.close();
         } catch (IOException ignored) {}
      }
   }
}

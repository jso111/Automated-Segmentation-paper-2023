import java.awt.image.BufferedImage;
import java.awt.image.RasterFormatException;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;



public class SplitLayer {

  
  
  private static int currentTiff = 1;

  private static String fileName;
  
  
  public static int numStacks = 0;
  
  
  
  public List<BufferedImage> extractImages(InputStream fileInput) throws Exception {
    List<BufferedImage> extractedImages = new ArrayList<BufferedImage>();

    try (ImageInputStream iis = ImageIO.createImageInputStream(fileInput)) {

        ImageReader reader = getTiffImageReader();
        reader.setInput(iis);

        int pages = reader.getNumImages(true);
        for (int imageIndex = 0; imageIndex < pages; imageIndex++) {
            BufferedImage bufferedImage = reader.read(imageIndex);
            extractedImages.add(bufferedImage);
        }
    }

    return extractedImages;
}

  private ImageReader getTiffImageReader() {
    Iterator<ImageReader> imageReaders = ImageIO.getImageReadersByFormatName("TIFF");
    if (!imageReaders.hasNext()) {
        throw new UnsupportedOperationException("No TIFF Reader found!");
    }
    return imageReaders.next();
}
  
  

  
  static boolean deleteDirectory(File directoryToBeDeleted) {
    File[] allContents = directoryToBeDeleted.listFiles();
    if (allContents != null) {
        for (File file : allContents) {
            deleteDirectory(file);
        }
    }
    return directoryToBeDeleted.delete();
}

public void listFilesForFolder(final File folder) {
  for (final File fileEntry : folder.listFiles()) {
      if (fileEntry.isDirectory()) {
          System.out.println(fileEntry.getName());
      }
  }
}
  
  public static void main(String[] args) {
    
    SplitLayer main = new SplitLayer();
    System.out.println("Running");
    Path currentRelativePath = Paths.get("");
    String s = currentRelativePath.toAbsolutePath().toString();
    File folder = new File(s+"/Images");
    if(!folder.exists()) {
      folder.mkdirs();
    }else {
      deleteDirectory(folder);
      folder.mkdirs();
    }
    System.out.println(s);

    File currentFolder = new File(s+"/Inputs");
    for(final File fileEntry : currentFolder.listFiles()){
      if(!fileEntry.isDirectory()){
        fileName = fileEntry.getName();
        File Outputs = new File(currentFolder+"/"+fileName.replaceFirst("[.][^.]+$", ""));
        Outputs.mkdir();
        System.out.println("Loop");
        try {
          System.out.println(fileName);
          List<BufferedImage> imageSet = main.extractImages(new FileInputStream(new File(currentFolder+"/"+fileName)));
          
          
        
          for(int j = 0; j<  imageSet.size(); j++) {
            //File saveImage = new File(imageSet.get(i));
            ImageIO.write(imageSet.get(j), "png", new File(Outputs.getAbsolutePath()+"/Layer"+(j+1)+".png"));
          }
        
        
      }
    

        catch(Exception e) {
          e.printStackTrace();
        }

        currentTiff++;
        System.out.println(currentTiff+", "+ "fileName");
    }
  }
    
    
  System.out.println("Done");
  }

}

'use client';

import { useState } from 'react';
import { Upload, FileVideo } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';

export default function VideoUploadPage() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [summary, setSummary] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoPreviewUrl(url);
      setSummary(null);
    }
  };

  const handleUpload = async () => {
    if (!videoFile) return;

    setIsProcessing(true);
    setSummary(null);

    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await fetch('/api/generate-summary', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to generate summary');
      }

      const data = await response.json();
      setSummary(data.summary);
    } catch (error) {
      console.error('Error generating summary:', error);
      setSummary('Error generating summary. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className='flex items-center justify-center min-h-screen p-4 bg-background'>
      <Card className='w-full max-w-5xl'>
        <div className='flex flex-col md:flex-row gap-6 p-6'>
          {/* Left side - Video upload */}
          <div className='flex flex-col gap-4 flex-1'>
            <h2 className='text-2xl font-bold'>Upload Video</h2>
            <div
              className={`border-2 border-dashed rounded-lg ${
                videoPreviewUrl ? 'border-border' : 'border-primary/50'
              } flex flex-col items-center justify-center p-6 h-[300px] relative`}
            >
              {videoPreviewUrl ? (
                <video
                  src={videoPreviewUrl}
                  controls
                  className='max-h-full max-w-full rounded'
                />
              ) : (
                <>
                  <FileVideo className='w-12 h-12 text-muted-foreground mb-4' />
                  <p className='text-muted-foreground text-center mb-2'>
                    Drag and drop your video here or click to browse
                  </p>
                  <p className='text-xs text-muted-foreground text-center'>
                    Supports MP4, WebM, and other common video formats
                  </p>
                </>
              )}
              <input
                type='file'
                accept='video/*'
                onChange={handleFileChange}
                className='absolute inset-0 w-full h-full opacity-0 cursor-pointer'
              />
            </div>
            <Button
              onClick={handleUpload}
              disabled={!videoFile || isProcessing}
              className='flex items-center gap-2'
            >
              <Upload className='w-4 h-4' />
              {isProcessing ? 'Processing...' : 'Generate Summary'}
            </Button>
          </div>

          <Separator
            orientation='vertical'
            className='hidden md:block h-auto'
          />

          <div className='flex flex-col gap-4 flex-1'>
            <h2 className='text-2xl font-bold'>Video Summary</h2>
            <div className='border rounded-lg p-4 h-[300px] overflow-y-auto'>
              {isProcessing ? (
                <div className='space-y-2'>
                  <Skeleton className='h-4 w-full' />
                  <Skeleton className='h-4 w-full' />
                  <Skeleton className='h-4 w-3/4' />
                  <Skeleton className='h-4 w-full mt-4' />
                  <Skeleton className='h-4 w-full' />
                  <Skeleton className='h-4 w-2/3' />
                </div>
              ) : summary ? (
                <div className='whitespace-pre-line'>{summary}</div>
              ) : (
                <div className='flex flex-col items-center justify-center h-full text-center text-muted-foreground'>
                  <p>
                    Upload a video and generate a summary to see the results
                    here
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

import { NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI('AIzaSyDv2g2QvGmdw4smcgZUBaIuth65ubqsQ-o');

export async function POST(request) {
  try {
    const formData = await request.formData();
    const videoFile = formData.get('video');

    if (!videoFile) {
      return NextResponse.json(
        { error: 'No video file provided' },
        { status: 400 }
      );
    }

    const bytes = await videoFile.arrayBuffer();
    const videoData = new Uint8Array(bytes);

    const fileType = videoFile.type;

    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

    const filePart = {
      inlineData: {
        data: Buffer.from(videoData).toString('base64'),
        mimeType: fileType,
      },
    };

    const result = await model.generateContent([
      `I’m uploading a video. Please watch the full video and provide a summary of the major events or important moments that occur.
      For each key event or turning point, include:
      A brief description of what happens
      The timestamp (in hh:mm:ss format) when the event starts
      Optionally, when it ends (if applicable)
      Make sure the summary is concise but informative, highlighting only the most relevant and meaningful parts of the video.`,
      filePart,
    ]);

    const response = await result.response;
    const summary = response.text().replaceAll('*', '');

    return NextResponse.json({ summary });
  } catch (error) {
    console.error('Error processing video:', error);
    return NextResponse.json(
      { error: 'Failed to process video' },
      { status: 500 }
    );
  }
}

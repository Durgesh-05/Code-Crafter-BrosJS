import { NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY);

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
      "Summarize the key events and information presented in the uploaded video.If there is any important frames make sure to brief it little.Don't give any type of punctuation.",
      filePart,
    ]);

    const response = await result.response;
    const summary = response.text();

    return NextResponse.json({ summary });
  } catch (error) {
    console.error('Error processing video:', error);
    return NextResponse.json(
      { error: 'Failed to process video' },
      { status: 500 }
    );
  }
}

import "./globals.css";

export const metadata = {
  title: "ClearComms",
  description: "Offline radio transcription to incident extraction"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="page-shell">{children}</div>
      </body>
    </html>
  );
}

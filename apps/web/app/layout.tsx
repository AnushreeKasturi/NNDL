import "./styles.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Legal Risk Classifier",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}


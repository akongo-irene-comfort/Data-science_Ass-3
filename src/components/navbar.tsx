"use client";

import { useState } from "react";
import Link from "next/link";
import { Menu, X, Traffic } from "lucide-react";

export const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const navLinks = [
    { href: "#overview", label: "Overview" },
    { href: "#problem", label: "Problem" },
    { href: "#algorithm", label: "Algorithm" },
    { href: "#simulation", label: "Simulation" },
    { href: "#implementation", label: "Implementation" },
    { href: "#deployment", label: "Deployment" },
    { href: "#demo", label: "Demo" },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-2 text-xl font-bold">
            <Traffic className="h-6 w-6 text-primary" />
            <span>RL Traffic Control</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-6">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
              >
                {link.label}
              </a>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden p-2"
            aria-label="Toggle menu"
          >
            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden py-4 space-y-2">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setIsOpen(false)}
                className="block px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors"
              >
                {link.label}
              </a>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
};

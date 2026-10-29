import { forwardRef, type ReactNode } from "react";
import { motion, type HTMLMotionProps } from "motion/react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { spring } from "@/lib/motion";

type Variant = "primary" | "ink" | "ghost" | "soft";

const variants: Record<Variant, string> = {
  primary: "bg-tomato text-paper border-ink shadow-pop hover:bg-tomato-deep",
  ink: "bg-ink text-paper border-ink shadow-[0_2px_0_0_var(--color-tomato)]",
  ghost: "bg-transparent text-ink border-ink/15 hover:border-ink/40 hover:bg-white/60",
  soft: "bg-white text-ink border-ink shadow-pop",
};

type Props = Omit<HTMLMotionProps<"button">, "children"> & {
  variant?: Variant;
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: ReactNode;
  children?: ReactNode;
};

export const PillButton = forwardRef<HTMLButtonElement, Props>(function PillButton(
  { variant = "primary", size = "md", loading, icon, className, children, disabled, ...rest },
  ref,
) {
  const sizes = {
    sm: "h-9 px-4 text-sm gap-1.5",
    md: "h-12 px-6 text-[0.95rem] gap-2",
    lg: "h-14 px-7 text-base gap-2.5",
  }[size];

  return (
    <motion.button
      ref={ref}
      whileHover={disabled || loading ? undefined : { y: -3, scale: 1.02 }}
      whileTap={disabled || loading ? undefined : { y: 1, scale: 0.96 }}
      transition={spring.bouncy}
      disabled={disabled || loading}
      className={cn(
        "relative inline-flex select-none items-center justify-center rounded-full border-2 font-semibold",
        "transition-colors duration-150 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-butter",
        "disabled:cursor-not-allowed disabled:opacity-55",
        sizes,
        variants[variant],
        className,
      )}
      {...rest}
    >
      {loading ? (
        <Loader2 className="size-4 animate-spin" aria-hidden />
      ) : (
        icon && (
          <motion.span
            className="inline-flex"
            whileHover={{ rotate: -12, scale: 1.15 }}
            transition={spring.bouncy}
          >
            {icon}
          </motion.span>
        )
      )}
      <span>{children}</span>
    </motion.button>
  );
});

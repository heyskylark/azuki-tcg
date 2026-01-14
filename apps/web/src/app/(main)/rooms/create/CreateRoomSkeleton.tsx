import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function CreateRoomSkeleton() {
  return (
    <div className="max-w-md mx-auto">
      <Card>
        <CardHeader>
          {/* CardTitle: leading-none font-semibold = ~24px */}
          <Skeleton className="h-6 w-28" />
          {/* CardDescription: text-sm = 20px line height */}
          <Skeleton className="h-5 w-80 max-w-full" />
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            {/* Label: text-sm font-medium = 20px */}
            <Skeleton className="h-5 w-36" />
            {/* Input: h-10 */}
            <Skeleton className="h-10 w-full" />
            {/* Helper text: text-sm = 20px */}
            <Skeleton className="h-5 w-72 max-w-full" />
          </div>
        </CardContent>
        <CardFooter>
          {/* Button: h-10 */}
          <Skeleton className="h-10 w-full" />
        </CardFooter>
      </Card>
    </div>
  );
}

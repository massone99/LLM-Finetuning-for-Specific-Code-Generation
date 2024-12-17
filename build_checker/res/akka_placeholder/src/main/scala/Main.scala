Here's a simple implementation of the `EscalatingSupervisor` actor in Scala, which supervises the `FaultyChildActor`. In this scenario, when the child actor throws a `NullPointerException` after receiving a "causeNull" message, the supervisor escalates the failure to its own supervisor.

```scala
// Define a supervisor strategy
trait SupervisorStrategy {
  def supervise(child: ActorRefAny): ActorRefAny

  def onChildException(child: ActorRefAny, exception: Exception): ActorRefAny =
    // This method should return the supervisor itself when a child actor fails
    // after throwing a specified exception (in this case, a NullPointerException).
    child? "causeNull" onErr (_ => supervisor)
}

// Define the EscalatingSupervisor actor
object EscalatingSupervisor {
  def main(args: Array[String]) {
    // Define supervisor strategy
    val supervisorStrategy = new SupervisorStrategy {
      def supervise(child: ActorRefAny): ActorRefAny = {
        println("Escalating failure to supervisor")
        // Return the supervisor actor
        ActorRefNoLogger(this)
      }
    }

    // Define child actor strategy
    val childStrategy = SupervisorStrategy {
      def supervise(child: ActorRefAny): ActorRefAny = {
        println("Faulty child actor receiving message causeNull...")
        // Throw a NullPointerException as an exception
        throw new NullPointerException("NullPointerException")
      }
    }

    // Create supervisor and child actor
    val supervisor = ActorSystem(supervisorStrategy.name).actorOf(supervisorStrategy.supervise, supervisorStrategy.name)
    val child = ActorSystem(supervisorStrategy.name).actorOf(supervisorStrategy.supervise, supervisorStrategy.name)

    // Test failure
    supervisor.tell("causeNull", Some(PoisonPill()))
  }
}
```

In the code above:

-   A custom supervisor strategy is defined which defines a behavior when the child actor throws an exception. The "onChildException" method will handle cases like the one described.
-   The "SupervisorStrategy" trait represents strategies to supervise child actors. Subclasses should implement this behavior to customize a child actor's failure.
-   The supervisor actor creates a child actor with a failing supervisor strategy. 
-   An attempt to pass a null value to the child actor triggers it to fail when receiving a "causeNull" message.

**Key concepts involved here:**

*   Actor supervision.
*   Supervisor strategies.
*   PoisonPill